# Third-party files

These files are not original to this project. Check each license before publishing the repo
publicly, and keep this list up to date.

| File | Source | License (verify) |
| --- | --- | --- |
| `face_detection_yunet_2023mar.onnx` | [OpenCV Zoo: YuNet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet) | MIT |
| `human_segmentation_pphumanseg_2023mar.onnx` | [OpenCV Zoo: PP-HumanSeg](https://github.com/opencv/opencv_zoo/tree/main/models/human_segmentation_pphumanseg) | Apache 2.0 |
| `haarcascade_frontalface_default.xml` | [OpenCV](https://github.com/opencv/opencv/tree/4.x/data/haarcascades) | Intel / OpenCV license (in the file header) |
| `Enochian.otf`, `EnochianPlain.ttf` | Unknown: add the font's source and license | **Unknown: remove before publishing if unclear** |

Presets also reference system fonts by path (`C:\Windows\Fonts\segoeui.ttf`,
`C:\Windows\Fonts\MATURASC.ttf`). Those are not included and must not be added: they are
licensed with Windows.
