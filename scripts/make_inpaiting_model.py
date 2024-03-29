import modules.scripts as scripts
import gradio as gr
import os
import torch
import tqdm
import safetensors.torch
import gc
import json

sdroot = "/".join(os.path.realpath(__file__).split("extensions")[0].split("/")[:-1])
models_folder_path = os.path.join(sdroot, "models/Stable-diffusion")

architecture_patterns = {
        "SDXL": "conditioner",
        "SD1": "cond_stage_model.transformer",
        "SD2": "cond_stage_model.model",
        "AltDiffusion": "cond_stage_model.roberta"
    }


def read_safetensors_header(filename: str) -> dict:
    error = f"{filename} не является корректным файлом .safetensors: "
    file_size = os.stat(filename).st_size
    if file_size < 8:
        raise ValueError(f"{error}размер менее 8 байт")

    with open(filename, "rb") as f:
        header_len = int.from_bytes(f.read(8), 'little', signed=False)
        if 8 + header_len > file_size:
            raise ValueError(f"{error}заголовок выходит за конец файла")

        hdr_buf = f.read(header_len)
        if len(hdr_buf) != header_len:
            raise ValueError(f"{error}размер заголовка {header_len}, но прочитано {len(hdr_buf)} байт")

        def parse_object_pairs(pairs):
            seen = set()
            for k, v in pairs:
                if k in seen:
                    raise ValueError(f"{error}дублирующиеся ключи в заголовке")
                seen.add(k)
            return dict(pairs)

        header = json.loads(hdr_buf, object_pairs_hook=parse_object_pairs)
        return header


def get_model_architecture(headers_dict: dict) -> str:
    for architecture, substring in architecture_patterns.items():
        if any(substring in key for key in headers_dict):
            return architecture
    return "Unknown"


def sd_version(safetensor_file: str) -> str:
    try:
        sf_headers = read_safetensors_header(safetensor_file)
        version = get_model_architecture(sf_headers)
        return version
    except Exception as e:
        return str(e)


def filter_sd1_files(directory: str) -> list:
    sd1_files = []
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path):
            version = sd_version(file_path)
            if version == "SD1":
                sd1_files.append(filename)
    return sd1_files


def check_path(path):
    if not path:
        return None
    if path is None:
        return None
    if path == "":
        return None
    if not os.path.exists(path):
        return None
    if not os.path.isfile(path):
        return None
    return path


def transform_checkpoint_dict_key(k, replacements):
    for text, replacement in replacements.items():
        if k.startswith(text):
            k = replacement + k[len(text):]
    return k


def get_state_dict_from_checkpoint(pl_sd):
    pl_sd = pl_sd.pop("state_dict", pl_sd)
    pl_sd.pop("state_dict", None)

    sd = {}
    for k, v in pl_sd.items():
        new_key = transform_checkpoint_dict_key(k, {
            'cond_stage_model.transformer.embeddings.': 'cond_stage_model.transformer.text_model.embeddings.',
            'cond_stage_model.transformer.encoder.': 'cond_stage_model.transformer.text_model.encoder.',
            'cond_stage_model.transformer.final_layer_norm.': 'cond_stage_model.transformer.text_model.final_layer_norm.',
        })

        if new_key is not None:
            sd[new_key] = v

    pl_sd.clear()
    pl_sd.update(sd)

    return pl_sd


def read_state_dict(checkpoint_file):
    map_location = str(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    if os.path.splitext(checkpoint_file)[1].lower() == ".safetensors":
        pl_sd = safetensors.torch.load_file(checkpoint_file, device=map_location)
    else:
        pl_sd = torch.load(checkpoint_file, map_location=map_location)
    return get_state_dict_from_checkpoint(pl_sd)


def to_half(tensor):
    if tensor.dtype == torch.float:
        return tensor.half()
    return tensor


def add_difference(theta0, theta1_2_diff, alpha):
    return theta0 + (alpha * theta1_2_diff)


def create_inpainting_model(model_a, model_b, model_c, custom_name):
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:24'
    model_a = check_path(model_a)
    model_b = check_path(model_b)
    model_c = check_path(model_c)
    if not model_a:
        raise ValueError("ОШИБКА: в папке моделей не найдена sd-v1-5-inpainting.safetensors")
    if not model_b:
        raise ValueError("ОШИБКА: для слияния нужно указать существующий путь до базовой модели.")
    if not model_c:
        raise ValueError("ОШИБКА: в папке моделей не найдена v1-5-pruned.safetensors")

    theta_1 = read_state_dict(model_b)
    theta_2 = read_state_dict(model_c)
    for key in tqdm.tqdm(theta_1.keys()):
        if key in ["cond_stage_model.transformer.text_model.embeddings.position_ids"]:
            continue
        if 'model' in key:
            if key in theta_2:
                t2 = theta_2.get(key, torch.zeros_like(theta_1[key]))
                theta_1[key] = theta_1[key] - t2
            else:
                theta_1[key] = torch.zeros_like(theta_1[key])
    del theta_2
    theta_0 = read_state_dict(model_a)
    for key in tqdm.tqdm(theta_0.keys()):
        if theta_1 and 'model' in key and key in theta_1:
            if key in ["cond_stage_model.transformer.text_model.embeddings.position_ids"]:
                continue

            a = theta_0[key]
            b = theta_1[key]

            if a.shape != b.shape and a.shape[0:1] + a.shape[2:] == b.shape[0:1] + b.shape[2:]:
                assert a.shape[1] == 9 and b.shape[1] == 4, f"неправильные размеры объединенного слоя {key}: A={a.shape}, B={b.shape}"
                theta_0[key][:, 0:4, :, :] = add_difference(a[:, 0:4, :, :], b, 1)
                result_is_inpainting_model = True
            else:
                theta_0[key] = add_difference(a, b, 1)
            theta_0[key] = to_half(theta_0[key])
    del theta_1

    safetensors.torch.save_file(theta_0, custom_name, None)

    del model_a
    del model_b
    del model_c
    del theta_0
    torch.cuda.empty_cache()
    gc.collect()


def make_inpainting_model(model_name: str) -> list:
    try:
        inpaint = os.path.join(models_folder_path, "sd-v1-5-inpainting.safetensors")
        base_sd = os.path.join(models_folder_path, "v1-5-pruned.safetensors")
        target_model = os.path.join(models_folder_path, model_name)
        output_model_path = os.path.join(models_folder_path, f"{os.path.splitext(model_name)[0]}-inpainting.safetensors")
        create_inpainting_model(inpaint, target_model, base_sd, output_model_path)
        return [f"модель сохранена: {output_model_path}", f"модель сохранена: {output_model_path}"]
    except Exception as e:
        msg = "нельзя использовать в качестве базы inpainting или instruct-pix2pix модель" if "size of tensor" in str(e) else str(e)
        return [msg, msg]


class Script(scripts.Script):

    def title(self):
        return "🎨️ создать inpaiting модель из базовой"

    def show(self, is_img2img):
        if is_img2img:
            return scripts.AlwaysVisible

    def ui(self, is_img2img):
        if is_img2img:
            with gr.Accordion(label=self.title(), open=True):

                sd1_file_paths = filter_sd1_files(models_folder_path)

                with gr.Row(variant="compact", elem_id="sd1_models"):
                    sd_model_list = gr.Dropdown(choices=sd1_file_paths, value=sd1_file_paths[0], label="", elem_id="sd_model_list")
                    refresh_button = gr.Button("🔄", variant="secondary", elem_id="refresh_inpainting_button")
                    create_inpainting_button = gr.Button("создать inpainting модель", variant="primary", elem_id="create_inpainting_button")

                def upd_models():
                    updated_sd1_file_paths = filter_sd1_files(models_folder_path)
                    return sd_model_list.update(choices=updated_sd1_file_paths, value=updated_sd1_file_paths[0])

                output_result = gr.HTML(value="", elem_id="inp_out")
                hidden_output = gr.Textbox(elem_id="hidden_inp_out")

                refresh_button.click(upd_models, outputs=sd_model_list)
                create_inpainting_button.click(make_inpainting_model, inputs=sd_model_list, outputs=[output_result, hidden_output])
                return [sd_model_list, refresh_button, create_inpainting_button, output_result]

