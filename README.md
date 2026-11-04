# Tennis Serve Analyzer

Estimates tennis serve speed from smartphone video, using a custom ball detector
and court geometry.

**Status:** working first pass on one clip. Speeds are rough estimates and have not
been checked against a radar gun.

## How it works

1. **Detect the ball.** YOLOv8n trained at 1280 px on a public dataset plus labeled
   court footage, because the ball is only about 25 px wide in a 1080p frame.
2. **Track it.** A Kalman filter rejects false detections and bridges short gaps.
3. **Find contact.** A sudden velocity change that follows a real toss.
4. **Find the bounce.** The detector usually loses the ball here, so frames are
   stabilized and differenced to follow it, and the bounce is where its descent and
   rise meet. If that evidence is missing, no speed is reported.
5. **Measure.** A court homography places the bounce in metres. Distance from contact
   to bounce, divided by flight time, gives average flight speed.

## Results

Two serves from one 240 fps handheld clip:

| Serve | Average flight speed | Estimated launch speed |
|---|---|---|
| First | 79 ± 3 km/h | ~95 km/h |
| Second | 90 ± 4 km/h | ~110 km/h |

Launch speed is a drag-model estimate, and is the figure closest to a radar reading.
The ± values cover input errors only. Automatic contact and bounce frames were within
1 frame of hand-marked ones. Detector mAP50 is 0.733.

## Limitations

- Not validated against radar.
- Event detection was tuned on these same two serves, and found no serves in a second
  clip where the server is far from the camera.
- Court landmarks and the ground point under contact are still marked by hand.
- One camera cannot directly measure depth, so this gives average speed, not true speed
  off the racket.

## Usage

Requires `ultralytics`, `opencv-python`, `numpy` and `matplotlib`. Footage and trained
weights are not included.

```
python track_ball.py --video clip.MOV --weights best.pt --out_dir out
python calibrate_court.py --video clip.MOV --frame 1014 --out out/court.json
python detect_events.py --track out/track.csv --video clip.MOV --out out/events.json
python estimate_speed.py --track out/track.csv --calibration out/court.json `
    --camera tripod --video clip.MOV --events out/events.json --serve 0 `
    --contact_ground_px X Y --out_dir out/serve0
```

Use `--camera handheld` for footage that was not shot on a tripod.

## Next

Record tripod footage from behind the server and side-on, test on serves not used for
tuning, add camera pose for true 3D positions, and validate against radar.

## Attribution

Base detection dataset: `tennisball-3eqxr/tennis-ball-detection-qaxae` from Roboflow
Universe, licensed CC BY 4.0.
