# 2025 - copyright - all rights reserved - clayton thomas baber

import asyncio
import pygame
import threading
import sys
import os
from cube import Cube
from model import RubikDistancePredictor
from torch import tensor, argsort, float32
from tkinter import Tk, filedialog
from collections import defaultdict

def pygame_loop(queue, stop_event):
    pygame.init()
    clock = pygame.time.Clock()  # Create a clock object
    my_font = pygame.font.SysFont('freesans', 100)

    display_info = pygame.display.Info()
    
    screen_width = display_info.current_w
    screen_height = display_info.current_h
    
    window_width = int(screen_width) // 3
    window_height = int(screen_height) // 3

    screen = pygame.display.set_mode((window_width, window_height))
    pygame.display.set_caption("Neural Cube")

    try:
        image_path = resource_path("assets/neuralcube.png")
        original_image = pygame.image.load(image_path) 
    except pygame.error as e:
        print(f"Error loading image: {e}")
        print(f"Image path: {image_path}")
        raise SystemExit(f"Could not load image: {e}")

    running = True
    solving = False
    stepping = False
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                stop_event.set()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                    stop_event.set()

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_1:
                    neuralcube.act(0)
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_2:
                    neuralcube.act(1)
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_3:
                    neuralcube.act(2)
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_4:
                    neuralcube.act(3)
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_5:
                    neuralcube.act(4)
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_6:
                    neuralcube.act(5)
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_7:
                    neuralcube.act(6)
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_8:
                    neuralcube.act(7)
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_9:
                    neuralcube.act(8)
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    neuralcube.act(9)
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_w:
                    neuralcube.act(10)
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_e:
                    neuralcube.act(11)
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    neuralcube.act(12)
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_t:
                    neuralcube.act(13)
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_y:
                    neuralcube.act(14)
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_u:
                    neuralcube.act(15)
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_i:
                    neuralcube.act(16)
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_o:
                    neuralcube.act(17)

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_HOME:
                    neuralcube.reset()
                    solving = False

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_END:
                    neuralcube.history = defaultdict(int)
                    solving = False

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_KP_ENTER:
                    if not neuralcube.isSolved():
                        solving = True

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_PERIOD:
                    stepping = True
            
        if solving or stepping:
            stepping = False

            state = neuralcube.getState()
            probe =  tensor(neuralcube.getProbe(), dtype=float32)
            predictions = model(probe).squeeze()
            choices = argsort(predictions)
            choice = neuralcube.history[state]

            if choice < 18:
                action = choices[choice]
                neuralcube.act(action)
                neuralcube.history[state] += 1
            else:
                solving = False
                neuralcube.history = defaultdict(int)

            if neuralcube.isSolved():
                solving = False
                neuralcube.history = defaultdict(int)                        

        result = neuralcube.toOneHot()
        result = tensor(result, dtype=float32)
        result = model(result)
        result = round(float(result) * 1000)
        result = str(result)

        image = pygame.Surface.copy(original_image)
        pixel_array = pygame.PixelArray(image)
        lookup = neuralcube.toColorHot(L=248)

        for sticker, chroma in enumerate(range(10, 64)):
            color_to_replace = (chroma, chroma, chroma)  
            new_color = lookup[sticker]
            pixel_array.replace(color_to_replace, new_color)
        pixel_array.close()

        image = pygame.transform.scale(image, (window_width, window_height))
        image_rect = image.get_rect(center = (window_width // 2, window_height // 2))
        
        text_surface = my_font.render(result, True, (0, 0, 0))

        screen.blit(image, image_rect)
        screen.blit(text_surface, (25, window_height - 115))

        pygame.display.flip()
        clock.tick(30)
    pygame.quit()

async def main():
    queue = asyncio.Queue()
    stop_event = asyncio.Event()

    pygame_thread = threading.Thread(target=pygame_loop, args=(queue, stop_event))
    pygame_thread.start()
    pygame_thread.join()
    print("finished")

def resource_path(relative_path):
        """ Get absolute path to resource, works for dev and for PyInstaller """
        try:
            # PyInstaller creates a temp folder and stores path in _MEIPASS
            base_path = sys._MEIPASS
        except Exception:
            base_path = os.path.abspath(".")

        return os.path.join(base_path, relative_path)


if __name__ == "__main__":
    
    neuralcube = Cube()
    neuralcube.history = defaultdict(int)

    neuralcube.algo(Cube.orbits[42])


    model_path = filedialog.askopenfilename(initialdir="lightning_logs")
    model = RubikDistancePredictor.load_from_checkpoint(model_path, map_location="cpu")

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Exiting due to keyboard interrupt.")