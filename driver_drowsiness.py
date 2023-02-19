#Importing OpenCV Library for basic image processing functions
import cv2
# Numpy for array related functions
import numpy as np
# Dlib for deep learning based Modules and face landmark detection
import dlib
#face_utils for basic operations of conversion
from imutils import face_utils
#Import pygame lib
import pygame
from pygame import mixer

import button

pygame.init()

Drowsy_Alertsound = mixer.Sound("D.wav")
Sleepy_Alertsound = mixer.Sound("S.wav")

#create game window
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600

screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Main Menu")

#game variables
game_paused = True

menu_state = "main"

#define fonts
font = pygame.font.SysFont("arialblack", 40)

#define colours
TEXT_COL = (255, 255, 255)

#load button images
resume_img = pygame.image.load("res/button_resume.png").convert_alpha()
quit_img = pygame.image.load("res/button_quit.png").convert_alpha()

#create button instances
resume_button = button.Button(304, 125, resume_img, 1)
quit_button = button.Button(336, 375, quit_img, 1)

def draw_text(text, font, text_col, x, y):
    img = font.render(text, True, text_col)
    screen.blit(img, (x, y))

def menu():
    run = True
    game_paused = True

    while run:

        screen.fill((52, 78, 91))

		#check if game is paused
        if game_paused == True:
            #check menu state
            if menu_state == "main":
                #draw pause screen buttons
                if resume_button.draw(screen):
                    game_paused = False
                if quit_button.draw(screen):
                    run = False
        else:
            StartApp()
            run = False

        #event handler
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

        pygame.display.update()

global drowsy
global sleep
global active
global status
global color

#Initializing the camera and taking the instance
cap = cv2.VideoCapture(0)

#Initializing the face detector and landmark detector
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

#status marking for current state
sleep = 0
drowsy = 0
active = 0
status=""
color=(0,0,0)

def compute(ptA, ptB):
	dist = np.linalg.norm(ptA - ptB)
	return dist

def blinked(a, b, c, d, e, f): #36, 37, 38, 41, 40, 39
	up = compute(b,d) + compute(c,e)
	down = compute(a,f)
	ratio = up/(2.0 * down)

	#Checking if it is blinked
	if (ratio > 0.25):
		return 2
	elif (ratio > 0.21 and ratio <= 0.25):
		return 1
	else:
		return 0

def lip (a, b, c, d):
	l = compute(a, d)
	r = compute(b, d)
	m = compute(c, d)

	#Checking if it is blinked
	if ((l > 2 * m) and (r > 2 * m)):
		return 1
	else:
		return 0

def StartApp():
	global drowsy
	global sleep
	global active
	global status
	global color

	while True:
		_, frame = cap.read()
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

		faces = detector(gray)
		for face in faces:
			x1 = face.left()
			y1 = face.top()
			x2 = face.right()
			y2 = face.bottom()

			face_frame = frame.copy()
			cv2.rectangle(face_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

			landmarks = predictor(gray, face)
			landmarks = face_utils.shape_to_np(landmarks)

			left_blink = blinked(landmarks[36], landmarks[37],
					landmarks[38], landmarks[41], landmarks[40], landmarks[39])
			right_blink = blinked(landmarks[42], landmarks[43],
					landmarks[44], landmarks[47], landmarks[46], landmarks[45])
			lip_motion = lip(landmarks[48], landmarks[54],
					landmarks[51], landmarks[57])
		
			if (left_blink == 0 and right_blink == 0 and lip_motion == 0):
				sleep += 1
				drowsy = 0
				active = 0
				if (sleep > 6):
					Drowsy_Alertsound.stop()
					Sleepy_Alertsound.play()
					status = "SLEEPING!!!"
					color = (255, 0, 0)
			elif (left_blink == 1 or right_blink == 1):
				sleep = 0
				active = 0
				drowsy += 1
				if (drowsy > 6):
					Sleepy_Alertsound.stop()
					Drowsy_Alertsound.play()
					status = "DROWSY!"
					color = (0, 0, 255)
			else:
				drowsy = 0
				sleep = 0
				active += 1
				if (active > 6):
					Sleepy_Alertsound.stop()
					Drowsy_Alertsound.stop()
					status = "Active"
					color = (0, 255, 0)
		
			cv2.putText(frame, status, (100,100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color,3)

			for n in range(0, 68):
				(x, y) = landmarks[n]
				cv2.circle(face_frame, (x, y), 1, (255, 255, 255), -1)
	
		cv2.imshow("Frame", frame)
		cv2.imshow("Result of detector", face_frame)
		key = cv2.waitKey(1)
		if key == 27:
			break

	cv2.destroyAllWindows()
	menu()

if __name__ == '__main__':
    menu()

pygame.quit()