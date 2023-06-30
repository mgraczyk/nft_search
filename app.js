// Setup the dnd listeners.

const el = s => document.getElementById(s);

const MOBILE_NET_INPUT_WIDTH = 224;
const MOBILE_NET_INPUT_HEIGHT = 224;

let mobilenet = null;

async function loadImageFromFile(file) {
  const url = URL.createObjectURL(file);
  try {
    return await new Promise((resolve, reject) => {
      const img = new Image();
      img.onload = () => {
        resolve(img);
      };
      img.onerror = () => {
        reject("Failed to load");
      };
      img.src = url;
    });
  } finally {
    URL.revokeObjectURL(url);
  }
}

async function loadMobileNetFeatureModel() {
  const URL = 
    'https://tfhub.dev/google/tfjs-model/imagenet/mobilenet_v3_small_100_224/feature_vector/5/default/1';
  
  mobilenet = await tf.loadGraphModel(URL, {fromTFHub: true});
  el('status').innerText = 'Model loaded, ready';
  
  // Warm up the model by passing zeros through it once.
  tf.tidy(() => {
    const answer = mobilenet.predict(tf.zeros([1, MOBILE_NET_INPUT_HEIGHT, MOBILE_NET_INPUT_WIDTH, 3]));
    console.log(answer.shape);
  });
}

const getFeatureVector = (image) => {
  const imageTensor = tf.browser.fromPixels(image);
  const resizedTensorFrame = tf.image.resizeBilinear(
    imageTensor, [MOBILE_NET_INPUT_HEIGHT, MOBILE_NET_INPUT_WIDTH], true);

  const imageFeatures = mobilenet.predict(resizedTensorFrame.expandDims()).arraySync();
  return imageFeatures[0];
}

// Get array buffers for each image, along with metadata.
const getImageFromPasteEvent = async (pasteEvent) => {
  const items = pasteEvent.clipboardData.items;
  console.log(items);

  const files = [];
  for (let i = 0; i < items.length; ++i) {
    const item = items[i];
    console.log(`item ${i}`, item);

    if (item.type.indexOf("image") < 0) {
      continue;
    }
    files.push(item.getAsFile());
  }
  if (files.length === 0) {
    console.error('no files to decode');
  }

  const file = files[0]

  const image = await loadImageFromFile(file);
  return image;
}

const isInputField = element => {
  const tagName = element.tagName.toLowerCase();
  return tagName === 'textarea' || tagName === 'input';
}

const startSearch = (image) => {
  const features = getFeatureVector(image)

  // After inference in case resizing changes things.
  image.style = 'width: 100%; height: 100%';
  el('query-img-container').replaceChildren(image);
  console.log('features', features);
  el('status').innerHTML = '<pre>Features: ' + features.map(x => x.toString()).join(',') + '</pre>';
};


// Call the function immediately to start loading.
loadMobileNetFeatureModel();

(function() {
  document.onpaste = async evt => {
    console.log('onpaste', evt);
    //if (isInputField(evt.target)) {
      //return;
    //}

    startSearch(await getImageFromPasteEvent(evt));
  };

  document.ondragover = evt => {
    evt.preventDefault();
  };

  document.ondrop = async evt => {
    if (isInputField(evt.target)) {
      return;
    }

    startSearch(await getImageFromPasteEvent(evt.clipboardData || window.clipboardData));
    evt.preventDefault();
  };

  const drop_zone = el('drop_zone');

  drop_zone.addEventListener('dragover', evt => {
    evt.stopPropagation();
    evt.preventDefault();
    evt.dataTransfer.dropEffect = 'copy';
  }, false);

  drop_zone.addEventListener('drop', evt => {
    evt.stopPropagation();
    evt.preventDefault();

    startSearch(evt.dataTransfer);
  }, false);

  el('image-form').addEventListener('onsubmit', evt => {
    evt.stopPropagation();
    evt.preventDefault();
    startSearch(evt);
  });
})();
