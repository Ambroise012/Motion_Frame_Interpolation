# Motion Frame Interpolation 
### Setup
```bash
pip install -r requirements.txt
```

Retrieve the folder containing all images of the animation : Streamboat Willie in [google drive](https://drive.google.com/drive/folders/18nWKoNI5MO-jutbhMQpcxeJYTZ8XiWl2?usp=sharing)

The final results are also avaible in the Drive.

## 1 - Horn-Schunck

```bash
python -m src.horn_schunck
```

Create movie from images + interpolated :
```bash
python make_movie.py \
  --image_folder output_horn \
  --output_video movies/anim_horn.mp4
```
## 2 - Farnebäck

```bash
python -m src.farneback
```

```bash
python make_movie.py \
  --image_folder output_farneback \
  --output_video movies/anim_farneback.mp4
```

## 3 - FILM: Frame Interpolation for Large Motion
```
git clone https://github.com/google-research/frame-interpolation
cd frame-interpolation
```

Download pre-trained TF2 Saved Models from
    [google drive](https://drive.google.com/drive/folders/1q8110-qp225asX3DQvZnfLfJPkCHmDpy?usp=sharing)
    and put into `<pretrained_models>`.

The downloaded folder should have the following structure:

```
<pretrained_models>/
├── film_net/
│   ├── L1/
│   ├── Style/
│   ├── VGG/
├── vgg/
│   ├── imagenet-vgg-verydeep-19.mat
```

```bash
python -m src.FILM
```
Make movies:
```bash
python make_movie.py \
  --image_folder output/output_film \
  --output_video movies/anim_film.mp4
```