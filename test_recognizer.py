#   USAGE
#   python test_recognizer.py --model ./datasets/fer2013/checkpoints/epoch_15.hdf5

from config import emotion_config as config
from pyimage.preprocessing.preprocessor import ImageToArrayPreprocessor
from pyimage.io.hdf5datasetgenerator import HDF5DatasetGenerator
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", type=str, help="path to model checkpoint to load")
args = vars(ap.parse_args())

testAug = ImageDataGenerator(rescale=1. / 255)
iap = ImageToArrayPreprocessor()

testGen = HDF5DatasetGenerator(config.TEST_HDF5,config.BATCH_SIZE,aug=testAug,preprocessor=iap,classes=config.NUM_CLASSES)

print("[INFO] loading {}...".format(args["model"]))
model = load_model(args["model"])

(loss,acc) = model.evaluate(
    testGen.generator(),
    steps=testGen.numImages
)
print("[INFO] loss: {}, acc: {}".format(loss,acc))
testGen.close()