import cv2
import numpy as np
import pygame
import sys

# Inicialización de Pygame
pygame.init()
pygame_screen = pygame.display.set_mode((640, 480))
pygame.display.set_caption('Simulated Robotic Arm')

# Inicialización de OpenCV
cap = cv2.VideoCapture(0)
lower_color = np.array([110, 50, 50])  # Supuesto rango bajo para azul
upper_color = np.array([130, 255, 255])  # Supuesto rango alto para azul

def detect_color(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_color, upper_color)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        max_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(max_contour)
        orientation = 'horizontal' if w > h else 'vertical'
        return (x, y, w, h), (x + w // 2, y + h // 2), orientation
    return None, None, None

def draw_pygame_scene(x, y, orientation):
    pygame_screen.fill((0, 0, 0))
    # Mesa y hueco
    pygame.draw.rect(pygame_screen, (0, 0, 255), (0, 400, 640, 80))  # Mesa
    pygame.draw.rect(pygame_screen, (100, 100, 100), (295, 400, 50, 80))  # Hueco
    # Restricción de altura para la pinza
    pinza_y = y if 295 <= x <= 345 else min(y, 370)  # Limita la altura para que no descienda sobre la mesa
    # Brazo y pinza
    pygame.draw.line(pygame_screen, (0, 255, 0), (320, 0), (320, pinza_y), 5)  # Brazo vertical
    pygame.draw.line(pygame_screen, (0, 255, 0), (320, pinza_y), (x, pinza_y), 5)  # Brazo horizontal
    if orientation == 'horizontal':
        pygame.draw.rect(pygame_screen, (0, 255, 0), (x - 25, pinza_y - 10, 50, 20))  # Pinza horizontal
    else:
        pygame.draw.rect(pygame_screen, (0, 255, 0), (x - 10, pinza_y - 25, 20, 50))  # Pinza vertical
    pygame.display.flip()

def main():
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        tracked_rect, tracked_pos, orientation = detect_color(frame)
        if tracked_rect:
            cv2.rectangle(frame, (tracked_rect[0], tracked_rect[1]), (tracked_rect[0] + tracked_rect[2], tracked_rect[1] + tracked_rect[3]), (0, 255, 0), 2)
            mapped_x = int(np.interp(tracked_pos[0], [0, frame.shape[1]], [0, 640]))
            mapped_y = int(np.interp(tracked_pos[1], [0, frame.shape[0]], [0, 480]))
            draw_pygame_scene(mapped_x, mapped_y, orientation)
        else:
            pygame_screen.fill((0, 0, 0))
            pygame.display.flip()

        cv2.imshow("Tracking", frame)
        for event in pygame.event.get():
            if event.type is pygame.QUIT:
                pygame.quit()
                cap.release()
                cv2.destroyAllWindows()
                sys.exit()

        if cv2.waitKey(1) & 0xFF == ord(' '):
            break

    pygame.quit()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

