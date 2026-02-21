import os
import io
import time
import threading
from pathlib import Path

import cv2
import numpy as np
import PySimpleGUI as sg
from PIL import Image

from ultralytics import YOLO


def list_builtin_models(models_dir: str):
    p = Path(models_dir)
    if not p.exists():
        return []
    return [str(x) for x in p.iterdir() if x.is_file() and x.suffix in ('.pt', '.onnx')]


def image_to_data(img: np.ndarray, maxsize=(800, 600)) -> bytes:
    if img is None:
        return b''
    if img.ndim == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(img)
    pil.thumbnail(maxsize)
    bio = io.BytesIO()
    pil.save(bio, format='PNG')
    return bio.getvalue()


class InferenceWorker(threading.Thread):
    def __init__(self, window, model_path, image_path, device):
        super().__init__(daemon=True)
        self.window = window
        self.model_path = model_path
        self.image_path = image_path
        self.device = device

    def run(self):
        start = time.time()
        try:
            model = YOLO(self.model_path)
            if self.device == 'gpu':
                device_arg = '0'
            else:
                device_arg = 'cpu'
            results = model.predict(source=self.image_path, device=device_arg, verbose=False)
            r = results[0]
            try:
                out_img = r.plot()
            except Exception:
                out_img = r.orig_img
            if out_img is None:
                raise RuntimeError('No output image from model')
            inference_time = time.time() - start
            text_lines = []
            text_lines.append(f'Device used: {"GPU" if device_arg != "cpu" else "CPU"}')
            text_lines.append(f'Model: {os.path.basename(self.model_path)}')
            text_lines.append(f'Inference time: {inference_time:.3f}s')
            names = []
            if hasattr(r, 'boxes') and r.boxes is not None:
                for box in r.boxes:
                    if hasattr(box, 'cls'):
                        try:
                            cls_idx = int(box.cls.tolist()[0])
                        except Exception:
                            cls_idx = None
                        if cls_idx is not None and hasattr(r, 'names'):
                            names.append(r.names.get(cls_idx, str(cls_idx)))
            if names:
                text_lines.append('Detected: ' + ', '.join(sorted(set(names))))
            else:
                text_lines.append('Detected: (none)')
            img_data = image_to_data(out_img)
            self.window.write_event_value('-INFERENCE_DONE-', (img_data, '\n'.join(text_lines)))
        except Exception as e:
            self.window.write_event_value('-INFERENCE_ERROR-', str(e))


def main():
    sg.theme('LightBlue')

    builtins = list_builtin_models(os.path.join(os.path.dirname(__file__), 'model'))

    layout = [
        [sg.Text('Image:'), sg.Input(key='-IMAGE_PATH-'), sg.FileBrowse(file_types=(('Image Files', '*.png;*.jpg;*.jpeg;*.bmp'),))],
        [sg.Text('Model:'), sg.Combo(values=builtins, key='-MODEL_COMBO-', size=(60,1)), sg.FileBrowse(button_text='Browse Model', key='-MODEL_BROWSE-', file_types=(('Model Files', '*.pt;*.onnx'),))],
        [sg.Radio('Auto (pref GPU)', 'DEVICE', default=True, key='-AUTO_DEVICE-'), sg.Radio('Force CPU', 'DEVICE', key='-FORCE_CPU-')],
        [sg.Button('Run Inference', key='-RUN-'), sg.Button('Exit')],
        [sg.HorizontalSeparator()],
        [sg.Column([[sg.Image(key='-OUTPUT_IMG-')]], vertical_alignment='top'), sg.VerticalSeparator(), sg.Multiline('', size=(40,25), key='-OUTPUT_TEXT-', disabled=True)],
    ]

    window = sg.Window('Breast Lesion Segmentation - GUI', layout, finalize=True)

    while True:
        event, values = window.read()
        if event in (sg.WIN_CLOSED, 'Exit'):
            break
        if event == '-RUN-':
            image_path = values.get('-IMAGE_PATH-')
            model_choice = values.get('-MODEL_COMBO-')
            browse_model = values.get('-MODEL_BROWSE-')
            model_path = browse_model or model_choice
            if not image_path or not os.path.exists(image_path):
                sg.popup('Please select a valid image file')
                continue
            if not model_path or not os.path.exists(model_path):
                sg.popup('Please select a valid model file (or pick one from the dropdown)')
                continue
            if values['-FORCE_CPU-']:
                device = 'cpu'
            else:
                device = 'gpu' if cv2.cuda.getCudaEnabledDeviceCount() > 0 else 'cpu'
            window['-OUTPUT_TEXT-'].update('Running inference...')
            window['-OUTPUT_IMG-'].update(data=b'')
            worker = InferenceWorker(window, model_path, image_path, device)
            worker.start()
        if event == '-INFERENCE_DONE-':
            img_data, text = values[event]
            window['-OUTPUT_IMG-'].update(data=img_data)
            window['-OUTPUT_TEXT-'].update(text)
        if event == '-INFERENCE_ERROR-':
            err = values[event]
            sg.popup('Inference error', str(err))

    window.close()


if __name__ == '__main__':
    main()
