import itertools
import os
import time
from glob import glob

import cv2
import numpy as np
from dataset import ImageFolder
from networks import *
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from utils_loss import *


class DCShadowNet(object):
    def __init__(self, args):
        self.model_name = "DCShadowNet"

        self.init_args(args)

        if torch.backends.cudnn.enabled and self.benchmark_flag:
            print("set benchmark !")
            torch.backends.cudnn.benchmark = True

        print()

        # print("##### Information #####")
        # print("# dataset : ", self.dataset)
        # print("# datasetpath : ", self.datasetpath)

    def init_args(self, args):
        self.modelpath = args.modelpath
        self.result_dir = args.result_dir
        self.dataset = args.dataset
        self.datasetpath = args.datasetpath

        self.iteration = args.iteration
        self.decay_flag = args.decay_flag

        self.batch_size = args.batch_size
        self.print_freq = args.print_freq
        self.save_freq = args.save_freq

        self.lr = args.lr
        self.weight_decay = args.weight_decay
        self.ch = args.ch
        self.step = args.step
        self.write_files = args.write_files

        """ Weight """
        self.adv_weight = args.adv_weight
        self.cycle_weight = args.cycle_weight
        self.identity_weight = args.identity_weight
        self.dom_weight = args.dom_weight

        self.use_crop = args.use_crop
        self.use_ch_loss = args.use_ch_loss
        self.use_pecp_loss = args.use_pecp_loss
        self.use_smooth_loss = args.use_smooth_loss

        if args.use_ch_loss == True:
            self.ch_weight = args.ch_weight
        if args.use_pecp_loss == True:
            self.pecp_weight = args.pecp_weight
        if args.use_smooth_loss == True:
            self.smooth_weight = args.smooth_weight

        """ Generator """
        self.n_res = args.n_res

        """ Discriminator """
        self.n_dis = args.n_dis

        self.img_size = args.img_size
        self.img_h = args.img_h
        self.img_w = args.img_w
        self.img_ch = args.img_ch

        self.device = args.device
        self.benchmark_flag = args.benchmark_flag
        self.resume = args.resume
        self.use_original_name = args.use_original_name
        self.im_suf_A = args.im_suf_A

    ##################################################################################
    # Model
    ##################################################################################

    def build_model(self):
        """DataLoader"""
        self.test_transform = transforms.Compose(
            [
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ]
        )

        """ Define Generator, Discriminator """
        self.genA2B = ResnetGenerator(
            input_nc=3,
            output_nc=3,
            ngf=self.ch,
            n_blocks=self.n_res,
            img_size=self.img_size,
            light=True,
        ).to(self.device)
        self.genB2A = ResnetGenerator(
            input_nc=3,
            output_nc=3,
            ngf=self.ch,
            n_blocks=self.n_res,
            img_size=self.img_size,
            light=True,
        ).to(self.device)
        self.disGA = Discriminator(input_nc=3, ndf=self.ch, n_layers=7).to(self.device)
        self.disGB = Discriminator(input_nc=3, ndf=self.ch, n_layers=7).to(self.device)
        self.disLA = Discriminator(input_nc=3, ndf=self.ch, n_layers=5).to(self.device)
        self.disLB = Discriminator(input_nc=3, ndf=self.ch, n_layers=5).to(self.device)

    def load(self):
        params = torch.load(self.modelpath)
        self.genA2B.load_state_dict(params["genA2B"])
        self.genB2A.load_state_dict(params["genB2A"])
        self.disGA.load_state_dict(params["disGA"])
        self.disGB.load_state_dict(params["disGB"])
        self.disLA.load_state_dict(params["disLA"])
        self.disLB.load_state_dict(params["disLB"])

    def transform_frame(self, frame):
        original_height, original_width = frame.shape[:2]
        # print(f'shape[:2]:{original_width}x{original_height}')
        original_aspect_ratio = original_width / original_height
        target_aspect_ratio = self.img_w / self.img_h

        # Переменные для обрезки
        x_start = y_start = 0

        # Переменные для добавления бордюров
        top_border = bottom_border = left_border = right_border = 0
        new_width = original_width
        new_height = original_height

        if self.use_crop:
            if original_aspect_ratio > target_aspect_ratio:
                # Обрезка по горизонтали
                new_width = int(original_height * target_aspect_ratio)
                x_start = (original_width - new_width) // 2
            else:
                # Обрезка по вертикали
                new_height = int(original_width / target_aspect_ratio)
                y_start = (original_height - new_height) // 2
        else:
            if original_aspect_ratio > target_aspect_ratio:
                # Сохранение ширины
                new_height = int(self.img_w / original_aspect_ratio)
                top_border = bottom_border = (self.img_h - new_height) // 2
            else:
                # Сохранение высоты
                new_width = int(self.img_h * original_aspect_ratio)
                left_border = right_border = (self.img_w - new_width) // 2

        if self.use_crop:
            cropped_frame = frame[
                y_start : y_start + new_height, x_start : x_start + new_width
            ]
            final_frame = cv2.resize(cropped_frame, (self.img_w, self.img_h))
        else:
            resized_frame = cv2.resize(frame, (new_width, new_height))
            final_frame = cv2.copyMakeBorder(
                resized_frame,
                top_border,
                bottom_border,
                left_border,
                right_border,
                cv2.BORDER_CONSTANT,
                value=[255, 255, 255],
            )

        return final_frame

    def insert_pad(self, frame, color=[255, 255, 255]):
        size = max(frame.shape[:2])
        padded_img = np.full((size, size, 3), color, dtype=np.uint8)

        x_offset = (size - frame.shape[1]) // 2
        y_offset = (size - frame.shape[0]) // 2

        padded_img[
            y_offset : y_offset + frame.shape[0],
            x_offset : x_offset + frame.shape[1],
        ] = frame

        return padded_img

    def remove_pad(self, frame, width, height):
        y_offset = (frame.shape[0] - height) // 2
        x_offset = (frame.shape[1] - width) // 2
        cropped_img = frame[y_offset : y_offset + height, x_offset : x_offset + width]

        return cropped_img

    def test_files(self, scale=0.25):
        target_path = self.result_dir
        if not os.path.exists(target_path):
            os.makedirs(target_path)

        # Получение списка PNG файлов в директории
        png_files = [f for f in os.listdir(self.datasetpath) if f.endswith(".png")]
        png_files.sort()

        for file_name in tqdm(png_files, desc="Обработка изображений"):
            frame = cv2.imread(os.path.join(self.datasetpath, file_name))
            frame = self.process_frame(frame, scale)

            cv2.imwrite(
                os.path.join(target_path, f"{file_name}{self.im_suf_A}"),
                frame,
            )

    def process_frame(self, frame, width, height):
        frame = cv2.resize(frame, (width, height))
        frame = self.insert_pad(frame)

        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        real_A = self.test_transform(img).unsqueeze(0).to(self.device)
        fake_A2B, _, _ = self.genA2B(real_A)
        B_fake = RGB2BGR(tensor2numpy(denorm(fake_A2B[0])))

        frame = (B_fake * 255).astype(np.uint8)
        frame = self.remove_pad(frame, width, height)
        return frame

    def test(self):
        self.load()
        print(" [*] Load SUCCESS")

        self.genA2B.eval()
        self.genB2A.eval()

        path_fakeB = os.path.join(self.result_dir, "output")
        if not os.path.exists(path_fakeB):
            os.makedirs(path_fakeB)

        video = cv2.VideoCapture(self.datasetpath)
        if not video.isOpened():
            print("Не удалось открыть видео.")
            return

        dataset_file = os.path.basename(self.datasetpath)

        # Определение параметров для VideoWriter
        fps = video.get(cv2.CAP_PROP_FPS)
        frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_video = cv2.VideoWriter(
            os.path.join(path_fakeB, dataset_file),
            fourcc,
            fps,
            (self.img_w, self.img_h),
        )

        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        for frame_number in tqdm(range(total_frames), desc="Обработка кадров"):
            ret, frame = video.read()
            if not ret:
                break

            if frame_number % self.step == 0:
                processed_frame = self.transform_frame(frame)

                # Преобразование обработанного кадра в формат, подходящий для модели
                img = Image.fromarray(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB))

                real_A = self.test_transform(img).unsqueeze(0).to(self.device)
                fake_A2B, _, _ = self.genA2B(real_A)
                B_fake = RGB2BGR(tensor2numpy(denorm(fake_A2B[0])))
                # cv2.imwrite(os.path.join(path_fakeB, f'frame_{frame_number:06}.png'), B_fake * 255.0)
                out_video.write((B_fake * 255).astype(np.uint8))

        video.release()
        out_video.release()
