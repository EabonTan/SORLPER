import math
import numpy as np
from batch_Parameters import env_params


class Env1(object):
    def __init__(self, em_seed):
        # self.agent = np.array([50, 50])
        self.agent = np.array(env_params['agent_start'])
        # self.friend = np.array([450, 450])
        self.friend = np.array(env_params['agent_goal'])
        self.selectEnv = env_params['env']
        np.random.seed(em_seed)
        self.generate_obstacles(self.selectEnv)

    # Library of geometric shape generation functions
    def generate_horizontal_line(self, y, x_range, step=10):
        """Generate obstacle points along a horizontal line."""
        return list(range(x_range[0], x_range[1] + 1, step)), [y] * len(range(x_range[0], x_range[1] + 1, step))

    def generate_vertical_line(self, x, y_range, step=10):
        """Generate obstacle points along a vertical line."""
        return [x] * len(range(y_range[0], y_range[1] + 1, step)), list(range(y_range[0], y_range[1] + 1, step))

    def generate_square(self, center, side_length, step=10):
        """Generate the boundary points of a complete square (including all vertices)."""
        cx, cy = center
        half = side_length // 2
        points_x = []
        points_y = []

        # Left vertical edge (including the upper and lower vertices)
        x_left = cx - half
        y_values = list(range(cy - half, cy + half + 1, step))
        if y_values[-1] != cy + half:  # Ensure that the endpoint is included
            y_values.append(cy + half)
        points_x.extend([x_left] * len(y_values))
        points_y.extend(y_values)

        # Upper horizontal edge (including the upper-right vertex)
        x_values = list(range(cx - half, cx + half + 1, step))
        if x_values[-1] != cx + half:  # Ensure that the endpoint is included
            x_values.append(cx + half)
        points_x.extend(x_values)
        points_y.extend([cy + half] * len(x_values))

        # Right vertical edge (including the lower-right vertex)
        x_right = cx + half
        y_values = list(range(cy + half, cy - half - 1, -step))
        if y_values[-1] != cy - half:  # Ensure that the endpoint is included
            y_values.append(cy - half)
        points_x.extend([x_right] * len(y_values))
        points_y.extend(y_values)

        # Lower horizontal edge (including the lower-left vertex)
        x_values = list(range(cx + half, cx - half - 1, -step))
        if x_values[-1] != cx - half:  # Ensure that the endpoint is included
            x_values.append(cx - half)
        points_x.extend(x_values)
        points_y.extend([cy - half] * len(x_values))

        return points_x, points_y

    def generate_triangle(self, center, side_length, step=10):
        """Generate the boundary points of an equilateral triangle."""
        cx, cy = center
        height = int(side_length * math.sqrt(3) / 2)
        points_x = []
        points_y = []

        # Bottom edge
        x_values = list(range(cx - side_length // 2, cx + side_length // 2, step))
        points_x.extend(x_values)
        points_y.extend([cy - height // 2] * len(x_values))

        # Left and right edges
        for direction in [-1, 1]:
            x_start = cx + direction * side_length // 2
            x_end = cx
            y_start = cy - height // 2
            y_end = cy + height // 2

            steps = abs(x_end - x_start) // step
            for i in range(steps + 1):
                x = x_start + (x_end - x_start) * i / steps
                y = y_start + (y_end - y_start) * i / steps
                points_x.append(int(x))
                points_y.append(int(y))

        return points_x, points_y

    def generate_circle(self, center, radius, num_points=36):
        """Generate the boundary points of a circle."""
        cx, cy = center
        points_x = []
        points_y = []

        for angle in range(num_points):
            rad = math.radians(360 * angle / num_points)
            x = cx + radius * math.cos(rad)
            y = cy + radius * math.sin(rad)
            points_x.append(int(x))
            points_y.append(int(y))

        return points_x, points_y

    def generate_pentagon(self, center, side_length, step=10):
        """Generate the boundary points of a regular pentagon."""
        cx, cy = center
        vertices = []
        for i in range(5):
            angle = math.radians(90 + 72 * i)  # Adjust the initial angle to make the bottom edge horizontal
            x = cx + side_length * math.cos(angle)
            y = cy + side_length * math.sin(angle)
            vertices.append((x, y))

        points_x = []
        points_y = []
        for i in range(5):
            x1, y1 = vertices[i]
            x2, y2 = vertices[(i + 1) % 5]
            dx = x2 - x1
            dy = y2 - y1
            distance = math.hypot(dx, dy)
            steps = int(distance / step)

            for t in range(steps + 1):
                x = x1 + dx * t / steps
                y = y1 + dy * t / steps
                points_x.append(int(x))
                points_y.append(int(y))

        return points_x, points_y

    def generate_hexagon(self, center, side_length, step=10):
        """Generate the boundary points of a regular hexagon."""
        cx, cy = center
        vertices = []
        for i in range(6):
            angle = math.radians(60 * i - 30)  # Adjust the initial angle to make the bottom edge horizontal
            x = cx + side_length * math.cos(angle)
            y = cy + side_length * math.sin(angle)
            vertices.append((x, y))

        points_x = []
        points_y = []
        for i in range(6):
            x1, y1 = vertices[i]
            x2, y2 = vertices[(i + 1) % 6]
            dx = x2 - x1
            dy = y2 - y1
            distance = math.hypot(dx, dy)
            steps = int(distance / step)

            for t in range(steps + 1):
                x = x1 + dx * t / steps
                y = y1 + dy * t / steps
                points_x.append(int(x))
                points_y.append(int(y))

        return points_x, points_y

        # # Generate example obstacle data
        # obstacles = {
        #     "Horizontal line": generate_horizontal_line(150, (50, 450), 15),
        #     "Vertical line": generate_vertical_line(300, (80, 420), 15),
        #     "Square": generate_square((200, 200), 100, 15),
        #     "Triangle": generate_triangle((350, 150), 120, 15),
        #     "Circle": generate_circle((150, 350), 60),
        #     "Hexagon": generate_hexagon((400, 350), 70, 15)
        # }

    def generate_obstacles(self, selectEnv):
        # Combine multiple geometric shapes to create a complex environment
        enx1, eny1 = self.generate_square((100, 400), 100)
        enx2, eny2 = self.generate_circle((400, 100), 50)
        enx3, eny3 = self.generate_hexagon((300, 300), 60)
        enx4, eny4 = self.generate_pentagon((150, 180), 50)
        # Merge the data
        if selectEnv == 1:
            combined_enx = enx1 + enx2 + enx3
            combined_eny = eny1 + eny2 + eny3
        elif selectEnv == 2:
            combined_enx = enx1 + enx2 + enx3 + enx4
            combined_eny = eny1 + eny2 + eny3 + eny4
        elif selectEnv == 0:
            combined_enx = enx3
            combined_eny = eny3
        self.ems = np.c_[combined_enx, combined_eny]

        self.d_em = 10
        self.d_fri = 10
        self.pre_fri_dis = np.linalg.norm((self.agent - self.friend))

    def step(self, action):
        # The action space contains 8 actions
        # change = [[0, -10], [0, 10], [-10, 0], [10, 0], [-10, 10], [-10, 10], [10, -10], [10, 10]]
        # self.agent = self.agent + change[action]
        _, _, pre_em_dis = self.get_observations()
        self.agent = self.agent + np.array(
            [int(10 * math.cos(action * np.pi / 18)), int(10 * math.sin(action * np.pi / 18))])

        state, fri_dis, cur_em_dis = self.get_observations()

        if (self.agent[0] < 10) or (self.agent[0] > 500) or (self.agent[1] < 10) or (self.agent[1] > 500) or (cur_em_dis < self.d_em):
            r1 = -50
        else:
            r1 = 0

        if fri_dis < self.d_fri:
            r2 = 100      # The goal reward is too small, causing the agent to favor obtaining r3
        else:
            r2 = 0

        if fri_dis < self.pre_fri_dis:
            r3 = 1
        else:
            r3 = -1
        # r3 = 0.1*(self.pre_fri_dis - fri_dis)

        if pre_em_dis < 25:
            if cur_em_dis < pre_em_dis:
                r4 = -0.5
            else:
                r4 = 1
        else:
            r4 = 0

        self.pre_fri_dis = fri_dis

        reward = r1 + r2 + r3 + r4

        if (self.agent[0] < 10) or (self.agent[0] > 500) or (self.agent[1] < 10) or (self.agent[1] > 500) or (
                cur_em_dis < self.d_em):
            done = True
            arrive = 0
        elif fri_dis < self.d_fri:
            done = True
            arrive = 1
        else:
            done = False
            arrive = 0

        info = {'agent': self.agent.reshape(1, 2), 'arrive': arrive}

        return state, reward, done, info

    def get_observation(self):
        self.friend = np.array(env_params['agent_goal'])
        # Calculate the distance from the agent to each obstacle
        em_s = np.linalg.norm(self.ems - self.agent, axis=1)  # Calculate the distance to each obstacle
        # Calculate the angle between each obstacle and the agent
        em_angles = np.arctan2(self.ems[:, 1] - self.agent[1], self.ems[:, 0] - self.agent[0])  # Calculate the angle
        # Find the minimum distance and its corresponding angle
        em_dis = np.min(em_s)  # Minimum distance
        idx = np.argmin(em_s)
        em_angel = em_angles[idx]

        fri_dis = np.linalg.norm((self.friend - self.agent))
        fri_angle = math.atan2((self.friend - self.agent)[1], (self.friend - self.agent)[0])

        state = np.array([np.cos(fri_angle), np.sin(fri_angle), np.cos(em_angel), np.sin(em_angel), fri_dis/(fri_dis+em_dis), em_dis/(fri_dis+em_dis)])

        return state, fri_dis, em_dis

    def get_observations(self):  # 12 static obstacles
        ems_s = np.linalg.norm((self.ems - self.agent), axis=1)
        idx = ems_s.argmin()
        em_dis = ems_s.min()
        em_angle = math.atan2((self.ems[idx, 1] - self.agent[1]), (self.ems[idx, 0] - self.agent[0]))

        fri_dis = np.linalg.norm((self.friend - self.agent))
        fri_angle = math.atan2((self.friend - self.agent)[1], (self.friend - self.agent)[0])

        # 6-dimensional state
        state = np.array([np.cos(fri_angle), np.sin(fri_angle), np.cos(em_angle), np.sin(em_angle),
                          fri_dis / (fri_dis + em_dis), em_dis / (fri_dis + em_dis)])
        return state, fri_dis, em_dis

    def reset(self):
        # self.agent = np.array([50, 50])
        self.agent = np.array(env_params['agent_start'])
        self.pre_fri_dis = np.linalg.norm((self.agent - self.friend))
        state, _, _ = self.get_observations()

        return state


if __name__ == '__main__':
    env = Env1()
    state = env.reset()
