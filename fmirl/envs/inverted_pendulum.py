# -*- coding: utf-8 -*-

__author__    = "Christian Richter"
__copyright__ = "Copyright 2019, TU Dresden"
__license__   = "GPL"
__credits__   = ["Christian Richter"]
__email__     = "christian.richter1@tu-dresden.de"
__project__   = "FmiRL"
__version__   = "0.1.0"


import os
import math
from fmirl import FmiEnvironment

# Define Parmeters
UPPER_X   =  2.4
LOWER_X   = -2.4
UPPER_PHI =  90+12
LOWER_PHI =  90-12
FMU_PATH  = os.path.join(os.path.dirname(__file__), "assets/inverted_pendulum/Pendel_Komponenten_Pendulum2.fmu")


class InvertedPendulum(FmiEnvironment):   
    def __init__(self):
        super().__init__(FMU_PATH)
        self.viewer = None
        self.reward_range = (0, 1)
        
    def initial_parameters(self):
        return {'m_trolley': 1, 'm_load': 0.1, 'phi1_start': math.radians(90 + self.random_state.uniform(-5, 5))}
    
    def calc_reward(self):
        return 1

    def is_done(self):
        done = False
        if self.state['phi1'] < math.radians(LOWER_PHI) or self.state['phi1'] > math.radians(UPPER_PHI):
            done = True
        
        if self.state['s'] < LOWER_X or self.state['s'] > UPPER_X:
            done = True
        return done
        

    def render(self, mode='human', close=False):
        # TODO: Rendering
        # Rendern sollte ein Live-Plot-Fenster öffnen, in dem die
        # aktuellen Werte der State-Variablen dargestellt werden.
        # z.B.: als Live-Graph oder Balkendiagramm
        # Bei Bedarf kann die Klasse in einer eigenen Umgebung überschrieben werden
        # z.B.: Zur Implementierung einer 3D-Visualisierung
        
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        screen_width = 600
        screen_height = 400

        world_width = 2.4 * 2
        scale = screen_width / world_width
        carty = 100  # TOP OF CART
        polewidth = 10.0
        polelen = scale * 1.0
        cartwidth = 50.0
        cartheight = 30.0

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height, display=None)
            l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
            axleoffset = cartheight / 4.0
            cart = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            self.carttrans = rendering.Transform()
            cart.add_attr(self.carttrans)
            self.viewer.add_geom(cart)
            l, r, t, b = -polewidth / 2, polewidth / 2, polelen - polewidth / 2, -polewidth / 2
            pole = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            pole.set_color(.8, .6, .4)
            self.poletrans = rendering.Transform(translation=(0, axleoffset))
            pole.add_attr(self.poletrans)
            pole.add_attr(self.carttrans)
            self.viewer.add_geom(pole)
            self.axle = rendering.make_circle(polewidth / 2)
            self.axle.add_attr(self.poletrans)
            self.axle.add_attr(self.carttrans)
            self.axle.set_color(.5, .5, .8)
            self.viewer.add_geom(self.axle)
            self.track = rendering.Line((0, carty), (screen_width, carty))
            self.track.set_color(0, 0, 0)
            self.viewer.add_geom(self.track)

        s = self.state['s']
        theta_ = self.state['phi1']

        cartx = s * scale + screen_width / 2.0  # MIDDLE OF CART
        self.carttrans.set_translation(cartx, carty)

        if theta_ > 0:
            theta_ -= 1.5707963267948966
        else:
            theta_ += 1.5707963267948966

        self.poletrans.set_rotation(-theta_)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')








