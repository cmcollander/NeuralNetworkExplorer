# A 2D Neural Network Explorer
# Copyright 2020 Chris Collander, cmcollander@gmail.com
# Licensed under GNU GPLv3 (LICENSE.md)

import pygame, pygame.locals, pygame.draw
import numpy as np
import sys
import time

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.activations import sigmoid, relu

PT_RADIUS = 8
PT_COLOR = ((255, 0, 0), (0, 0, 255))
ORIG_BACKGROUND_COLOR = (0, 50, 0)
BACKGROUND_COLOR = ((50, 0, 0), (0, 0, 50))
TEXT_COLOR = (255, 255, 255)


def _map(x, in_min, in_max, out_min, out_max):
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min


def normalize(x, y):
    return _map(x, 0.0, 800.0, -1.0, 1.0), _map(y, 0.0, 600.0, -1.0, 1.0)


def unnormalize(x, y):
    return int(_map(x, -1.0, 1.0, 0.0, 800.0)), int(_map(y, -1.0, 1.0, 0.0, 600.0))


# Get Pygame ready!
pygame.init()
DISPLAYSURF = pygame.display.set_mode((800, 600), pygame.DOUBLEBUF | pygame.HWSURFACE)
pygame.display.set_caption("2D Neural Network Explorer!")
sysfont = pygame.font.get_default_font()
font = pygame.font.SysFont(None, 24)
font_img_1 = font.render("Left click to place Red points", True, TEXT_COLOR)
font_img_2 = font.render("Right click to place Blue points", True, TEXT_COLOR)
font_img_3 = font.render("Hit Enter to train and view results", True, TEXT_COLOR)

update_display = True
new_model = True
first_model = True

data = []

# Adjust this model to anything you want!!
num_epochs = 2000
batch_size = 8
validation_split = 0.2
model = Sequential()
model.add(Dense(4, input_shape=(2,), activation=relu))
model.add(Dense(4, activation=relu))
model.add(Dense(1, activation=sigmoid))
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
model.save_weights("model.h5")

coord_list = []
coord_list_norm = []
coord_map = {}
for x in range(800):
    for y in range(600):
        coord_list.append((x, y))
        coord_list_norm.append(normalize(x, y))
        coord_map[(x, y)] = len(coord_list) - 1

while True:
    event = pygame.event.poll()
    if event.type == pygame.locals.MOUSEBUTTONDOWN:
        label = 0 if event.button == 1 else 1
        x = event.pos[0]
        y = event.pos[1]
        new_x, new_y = normalize(x, y)
        data.append([new_x, new_y, label])
        print(f"Received {len(data)} points so far!")
        update_display = True
    if event.type == pygame.locals.KEYDOWN and event.key == pygame.K_RETURN:
        if len(data) < 10:
            print(
                "Wait until you have at least 10 points before trying to train a model!"
            )
        else:
            print("Training! Please wait...")
            # Prepare data for training
            start = time.time()
            X = np.array(data)[:, :2]
            Y = np.array(data)[:, -1]
            # Let's recover our initial model, so we aren't biased by previous trainings
            model.load_weights("model.h5")
            # And TRAIN!
            model.fit(
                X,
                Y,
                epochs=num_epochs,
                batch_size=batch_size,
                verbose=0,
                validation_split=validation_split,
            )
            end = time.time()
            print(f"Training time: {end-start}")
            # Evaluate the keras model
            _, accuracy = model.evaluate(X, Y)
            print("Accuracy: %.2f" % (accuracy * 100))
            update_display = True
            new_model = True
            first_model = False
    if event.type == pygame.locals.QUIT:
        pygame.quit()
        sys.exit()

    # If we don't need to update the display, back out at this point!
    if not update_display:
        continue

    # If a model has been trained, predict EVERY point in space and show us what it will be assigned
    # But ONLY if this model differs from our last
    if new_model:
        DISPLAYSURF.fill(ORIG_BACKGROUND_COLOR)
        if not first_model:
            print("Predicting all background pixels! Please wait...")
            start = time.time()
            res = model.predict(coord_list_norm)
            for coord in coord_list:
                pixel_res = res[coord_map[coord]][0]
                color = BACKGROUND_COLOR[1 if pixel_res > 0.5 else 0]
                DISPLAYSURF.set_at(coord, color)
            end = time.time()
            print(f"Inference time: {end-start}")
            new_model = False
    # display all points
    for row in data:
        pt = row[:2]
        new_pt = unnormalize(pt[0], pt[1])
        color = PT_COLOR[row[-1]]
        pygame.draw.circle(DISPLAYSURF, color, new_pt, PT_RADIUS)
    DISPLAYSURF.blit(font_img_1, (20, 520))
    DISPLAYSURF.blit(font_img_2, (20, 550))
    DISPLAYSURF.blit(font_img_3, (20, 580))
    pygame.display.update()
