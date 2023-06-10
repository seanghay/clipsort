import os
from argparse import ArgumentParser
from pathlib import Path
from transformers import pipeline
from PIL import Image
from tqdm import tqdm
import shutil

def processor(classifier, src_path, labels, out_dir, take, overwrite=False):
 
  with open(src_path, "rb") as fp:
    image = Image.open(fp)
    results = classifier(image, candidate_labels=labels)
  
    for result in results[0:take]:
      label = result['label']
      dest_path = os.path.join(out_dir, label, src_path.name)
  
      if not os.path.isfile(dest_path) or overwrite:
        shutil.copyfile(src_path, dest_path)

def main():
  # parse args
  parser = ArgumentParser("clipsort",description="Sort unstructured photos with CLIP")
  parser.add_argument("files", nargs="+", type=Path, default=[], help="Path to files")
  parser.add_argument("-o", "--output", type=Path, default=Path("output"), help="Output folder")
  parser.add_argument("-l", "--labels", nargs="+", required=True)
  parser.add_argument("-w", "--overwrite", action="store_true", default=True, help="Overwrite exsting file in the output dir")
  parser.add_argument("-t", "--take", type=int, default=1, help="Number of final labels to save")
  parser.add_argument("-m", "--model", default="openai/clip-vit-base-patch16")

  args = parser.parse_args()
  
  # num of labels must be greater than 1 otherwise it makes no sense
  assert len(args.labels) > 1
  assert args.take < len(args.labels)

  # load the model
  model = args.model
  classifier = pipeline("zero-shot-image-classification", model=model)

  # create the output folder
  for label in args.labels: 
    label_dir = os.path.join(args.output, label)
    os.makedirs(label_dir, exist_ok=True)

  for src_path in tqdm(args.files):
    processor(
      classifier=classifier, 
      src_path=src_path, 
      labels=args.labels,
      out_dir=args.output,
      take=args.take,
      overwrite=args.overwrite,
    )

if __name__ == "__main__":
  main()