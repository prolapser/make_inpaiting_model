onUiLoaded(function () {
  var createInpaintingModelButton = document.getElementById('create_inpainting_button');
  var MakeInpaintHiddenOutput = document.querySelector('#hidden_inp_out');
  
  function updateMakeInpainModelButtonState() {
    var hiddenInpOutFirstChild = document.querySelector('#hidden_inp_out > div:first-child');
    createInpaintingModelButton.disabled = hiddenInpOutFirstChild && hiddenInpOutFirstChild.children.length > 0;
  }
  var InpaintingModelOutputObserver = new MutationObserver(function(mutations) {
    mutations.forEach(function(mutation) {
      if (mutation.addedNodes.length > 0 || mutation.removedNodes.length > 0) {
        updateMakeInpainModelButtonState();
      }
    });
  });
  
  if (MakeInpaintHiddenOutput) {
    InpaintingModelOutputObserver.observe(MakeInpaintHiddenOutput, { childList: true, subtree: true });
  }
  
  styles = `
  #sd1_models {
    display: flex;
    flex-direction: row;
    flex-wrap: nowrap;
    align-content: center;
    align-items: center;
    justify-content: space-between;
}

#refresh_inpainting_button {
    --s: 18px;
    max-width: var(--s) !important;
    width: var(--s) !important;
    min-width: var(--s) !important;
    max-height: var(--s) !important;
    height: var(--s) !important;
    min-height: var(--s) !important;
    background: none!important;
    border: none!important;
    padding: 0!important;
    margin: 0!important;
    display: flex;
    flex-wrap: wrap;
    align-content: center;
    justify-content: center;
    align-items: center;
}
#sd_model_list span {
    display: none !important;
    width: 0px !important;
    height: 0px !important;
}

#hidden_inp_out {
    display: none
}
@keyframes moving-stripes {
    0% {
        background-position: 0 0;
    }
    100% {
        background-position: 50px 0;
    }
}

#create_inpainting_button[disabled] {
    cursor: progress;
    background-size: 50px 50px;
    background-image: linear-gradient( 45deg,
    hsl(0deg 0% 50% / 50%) 25%,
    transparent 25%,
    transparent 50%,
    hsl(0deg 0% 50% / 50%) 50%,
    hsl(0deg 0% 50% / 50%) 75%,
    transparent 75%,
    transparent) !important;
    animation: moving-stripes 2s linear infinite;
    color: transparent;
    transition: 0.5s !important;
}
  `
  const BodyStyle = document.createElement('style');
  BodyStyle.setAttribute("data-name", "create_inpaiting")
  document.querySelector("body").appendChild(BodyStyle);
  BodyStyle.insertAdjacentHTML("beforeend", styles);
  });
