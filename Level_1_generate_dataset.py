"""At level 1, it spawns a ball on three randomly placed platforms
"""

__version__ = "$Id:$"
__docformat__ = "reStructuredText"

# Python imports
import random
import math
import numpy as np

# Library imports
import pygame
from pygame.key import *
from pygame.locals import *
from pygame.color import *

# pymunk imports
import pymunk
import pymunk.pygame_util

import torch


class BouncyBalls(object):
    """
    This class implements a simple scene in which there is a static platform (made up of a couple of lines)
    that don't move. Balls appear occasionally and drop onto the platform. They bounce around.
    """
    def __init__(self, data_folder):
        # Space
        self._space = pymunk.Space()
        self._space.gravity = (0.0, -900.0)

        # Physics
        # Time step
        self._dt = 1.0 / 60.0
        # Number of physics steps per screen frame
        self._physics_steps_per_frame = 1

        # pygame
        pygame.init()
        self._screen = pygame.display.set_mode((600, 600))
        self._clock = pygame.time.Clock()

        self._draw_options = pymunk.pygame_util.DrawOptions(self._screen)

        # Static lines that exist in the world
        self._static_lines = []
        
        # Balls that exist in the world
        self._balls = []

        # Execution control and time until the next ball spawns
        self._running = True
        self._ticks_to_next_ball = 10
        self._ticks_to_next_line = 10
        self.episode_len = 150
        self.data_folder_path = data_folder#'/hdd_d/yukun/SWIM/IN/dataset/level_2/'
        self.episode_idx = 0
        self.episode_buffer = []

    def run(self):
        """
        The main loop of the game.
        :return: None
        """
        # Main loop
        while self._running:
            # Progress time forward
            for x in range(self._physics_steps_per_frame):
                self._space.step(self._dt)

            self._process_events()
            
            #self._update_balls() # use self._update_scenery() instead Y.D.
            #self._update_static_scenery() # use self._update_scenery() instead Y.D.
            
            self._update_scenery()
            
            self._clear_screen()
            self._draw_objects()
            self._save_state_data()
            pygame.display.flip()
            # Delay fixed time between frames
            self._clock.tick(50)
            pygame.display.set_caption("fps: " + str(self._clock.get_fps()))

    def _get_end_point(self, x, y, theta, length=70):
        """Given start point, angle, and length, return the end point
        """
        return x+math.cos(theta)*length, y+math.sin(theta)*length

    def _save_state_data(self):
        """Save episode information to file.
        """
        if len(self._balls) == 1:
            # save if episode ends
            if self._ticks_to_next_ball == self.episode_len:
                if len(self.episode_buffer) > 0:
                    assert len(self.episode_buffer) == self.episode_len, 'data length not equal to episode length'
                    np.save(self.data_folder_path+str(self.episode_idx)+'.npy', np.array(self.episode_buffer))
                    self.episode_idx += 1
                    self.episode_buffer = []
            # collect information
            ball_body = self._balls[0].body
            step_data = [ball_body.angle, ball_body.angular_velocity, ball_body.position.x, ball_body.position.y, ball_body.velocity.x, ball_body.velocity.y] + self.line_property
            self.episode_buffer.append(np.array(step_data))    
    
    def _add_static_scenery(self):
        """
        Create the static bodies.
        :return: None
        """
        # a static body to be used
        body = pymunk.Body(body_type = pymunk.Body.STATIC) # 1
        body.position = (0, 0)
        static_body = body
        
        # set the platforms
        x_s_1 = 30.0+random.randint(-25,25)
        y_s_1 = 270.0+random.randint(-25,25)
        theta_1 = -0.25*math.pi
        x_e_1, y_e_1 = self._get_end_point(x_s_1, y_s_1, theta_1, 90)
        
        x_s_2 = 200.0+random.randint(-25,25)
        y_s_2 = 200.0+random.randint(-25,25)
        theta_2 = 0
        x_e_2, y_e_2 = self._get_end_point(x_s_2, y_s_2, theta_2, 70)
        
        x_s_3 = 350.0+random.randint(-25,25)
        y_s_3 = 200.0+random.randint(-25,25)
        theta_3 = 0
        x_e_3, y_e_3 = self._get_end_point(x_s_3, y_s_3, theta_3, 80)
        
        self.line_property = [x_s_1, y_s_1, theta_1, x_s_2, y_s_2, theta_2, x_s_3, y_s_3, theta_3]
        
        static_lines = [pymunk.Segment(static_body, (x_s_1, y_s_1), (x_e_1, y_e_1), 2.0),
                        pymunk.Segment(static_body, (x_s_2, y_s_2), (x_e_2, y_e_2), 2.0),
                        pymunk.Segment(static_body, (x_s_3, y_s_3), (x_e_3, y_e_3), 2.0)
                       ]
        
        static_lines[0].elasticity = 0.7
        static_lines[1].elasticity = 0.9
        static_lines[2].elasticity = 0.8
        for line in static_lines:
            line.friction = 0.04
        self._static_lines = static_lines
        self._space.add(static_lines)

    def _process_events(self):
        """
        Handle game and events like keyboard input. Call once per frame only.
        :return: None
        """
        for event in pygame.event.get():
            if event.type == QUIT:
                self._running = False
            elif event.type == KEYDOWN and event.key == K_ESCAPE:
                self._running = False
            elif event.type == KEYDOWN and event.key == K_p:
                pygame.image.save(self._screen, "bouncing_balls.png")

    def _update_balls(self):
        """
        Create/remove balls as necessary. Call once per frame only.
        :return: None
        """
        self._ticks_to_next_ball -= 1
        if self._ticks_to_next_ball <= 0:
            self._create_ball()
            self._ticks_to_next_ball = self.episode_len
        # Remove balls that fall below 100 vertically
        balls_to_remove = [ball for ball in self._balls if ball.body.position.y < 100]
        for ball in balls_to_remove:
            self._space.remove(ball, ball.body)
            self._balls.remove(ball)
            
    def _update_scenery(self):
        """
        Create/remove static bodies as necessary. Call once per frame only.
        :return: None
        Y.D.
        """
        self._ticks_to_next_ball -= 1
        # refresh
        if self._ticks_to_next_ball <= 0:
            # remove platforms
            self._space.remove(self._static_lines)
            # remove ball
            for ball in self._balls:
                self._space.remove(ball, ball.body)
                self._balls.remove(ball)
            # create ball
            self._create_ball()
            # create platforms
            self._add_static_scenery()
            # reset ticks
            self._ticks_to_next_ball = self.episode_len

    def _create_ball(self):
        """
        Create a ball.
        :return:
        """
        mass = 10
        radius = 10
        inertia = pymunk.moment_for_circle(mass, 0, radius, (0, 0))
        body = pymunk.Body(mass, inertia)
        x = 60
        body.position = x, 450
        shape = pymunk.Circle(body, radius, (0, 0))
        shape.elasticity = 0.95
        shape.friction = 0.9
        self._space.add(body, shape)
        self._balls.append(shape)

    def _clear_screen(self):
        """
        Clears the screen.
        :return: None
        """
        self._screen.fill(THECOLORS["white"])

    def _draw_objects(self):
        """
        Draw the objects.
        :return: None
        """
        self._space.debug_draw(self._draw_options)

import argparse

def get_args():
    parser = argparse.ArgumentParser(description='run simulation and generate dataset',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d', '--data_folder', dest='folder', type=str, default='debug',
                        help='save data to the data folder')
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    game = BouncyBalls(args.folder)
    game.run()
