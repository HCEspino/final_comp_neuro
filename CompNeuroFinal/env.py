from spikeWaveEprop import update_weights
import numpy as np
import copy

map_size_x = 10
map_size_y = 10

environment = np.array([[5, 5, 5, 5, 1, 1, 5, 5, 5, 5],
                        [5, 5, 5, 5, 1, 1, 5, 5, 5, 5],
                        [5, 5, 5, 15, 1, 1, 15, 5, 5, 5],
                        [5, 5, 15, 15, 1, 1, 15, 15, 5, 5],
                        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                        [5, 5, 15, 15, 1, 1, 15, 15, 5, 5],
                        [5, 5, 5, 15, 1, 1, 15, 5, 5, 5],
                        [5, 5, 5, 5, 1, 1, 5, 5, 5, 5],
                        [5, 5, 5, 5, 1, 1, 5, 5, 5, 5]])
#long_env = np.concatenate((environment, environment[0:5, :]))
#big_env = np.concatenate((long_env, long_env[:, 0:5]), axis=1)
#environment = big_env

def softmax(x, temperature=1):
    return np.exp(np.array(x)/temperature)/sum(np.exp(np.array(x)/temperature))

class Buffer():
    def __init__(self, maxlen):
        self.stg = []
        self.maxlen = maxlen
    
    def push(self, item):
        self.stg.append(item)
        if len(self.stg) > self.maxlen:
            del self.stg[-1]

    def get(self, idx):
        return self.stg[idx]

    def length(self):
        return len(self.stg)

    def set(self, idx, item):
        self.stg[idx] = item

class Agent():
    def __init__(self, pos, memorylen, y, x):
        self.current_pos = pos

        self.y = y
        self.x = x
        self.expmap = np.ones((y, x)).astype('float32')

        self.memorylen = memorylen
        self.etmem = Buffer(memorylen)
        self.pathmem = Buffer(memorylen)

    def reset_exp(self):
        #self.expmap = np.ones((self.y, self.x)).astype('float32')
        self.expmap *= 1.10
        self.expmap = np.clip(self.expmap, 0, 1)

    def getexpscores(self):
        p = []
        for i in range(self.pathmem.length()):
            avg_score = 0
            count = 0
            path = self.pathmem.get(i)
            for point in path:
                avg_score += self.expmap[point[0]][point[1]]
                count += 1
            
            avg_score = avg_score / count
            p.append(avg_score)
        return p

    def replay(self, n, cost_map, wgt, type):

        #Current solution: Memory access is chosen based on distance between current position and average of path's position. Scaled based on amount of times it's been accessed. Doesn't perform better than completely uniform.

        if type == "exp":

            p = self.getexpscores()
            p_sm = softmax(p, 10)

            for i in range(n):
                idx = np.random.choice(list(range(self.pathmem.length())), p=p_sm)
                path = self.pathmem.get(idx)
                et = self.etmem.get(idx)[0]
                wgt = update_weights(cost_map, et, path, wgt)

                #Decay replay accessed, reassess softmax
                avg_score = 0
                count = 0
                for point in path:
                    self.expmap[point[0]][point[1]] *= 0.75

                p = []
                for i in range(self.pathmem.length()):
                    avg_score = 0
                    count = 0
                    path = self.pathmem.get(i)
                    for point in path:
                        avg_score += self.expmap[point[0]][point[1]]
                        count += 1
                    
                p = self.getexpscores()
                p_sm = softmax(p, 10)


                p_sm = softmax(p, 10)

        elif type == "recent":
            #Uniform distribution
            #p = [self.pathmem.length() - i for i in range(self.pathmem.length())]

            #p = [5 - np.exp(0.025*i) for i in range(self.pathmem.length())]
            p = [self.pathmem.length() - i for i in range(self.pathmem.length())]

            #Scale probabiliies by amount of times accessed (no scaling right now)
            for j in range(len(p)):
                p[j] = p[j] * self.etmem.get(j)[1]

            p_sm = softmax(p, 25.0)

            for i in range(n):
                idx = np.random.choice(list(range(self.pathmem.length())), p=p_sm)
                path = self.pathmem.get(idx)
                et = self.etmem.get(idx)[0]
                wgt = update_weights(cost_map, et, path, wgt)

                #Decay replay accessed, reassess softmax
                self.etmem.set(idx, (self.etmem.get(idx)[0], self.etmem.get(idx)[1] * 0.50))
                p[idx] *= 0.5
                p_sm = softmax(p, 25.0)       

    #    if type == "dist":
            #Choose memory probability based on distance between current location and average distance of path
    #        p = []
    #        for i in range(self.pathmem.length()):
    #            avg_dist = 0
    #            count = 0
    #           path = self.pathmem.get(i)
    #            for point in path:
    #                avg_dist += np.sqrt((float(self.current_pos[0]) - point[0])**2 + (float(self.current_pos[1]) - point[1])**2)
    #                count += 1

    #            avg_dist = avg_dist / count
    #            p.append(avg_dist)

            #Scale probabiliies by amount of times accessed
    #        for j in range(len(p)):
    #            p[j] = p[j] * self.etmem.get(j)[1]

    #        p_sm = softmax(p, 1.5)

    #        for i in range(n):
    #            idx = np.random.choice(list(range(self.pathmem.length())), p=p_sm)
    #            path = self.pathmem.get(idx)
    #            et = self.etmem.get(idx)[0]
    #            wgt = update_weights(cost_map, et, path, wgt)

                #Decay replay accessed, reassess softmax
    #            self.etmem.set(idx, (self.etmem.get(idx)[0], self.etmem.get(idx)[1] * 0.50))
    #            p[idx] *= 0.50
    #            p_sm = softmax(p, 1.5)

        else:
            #Uniform distribution
            p = [1/self.pathmem.length() for i in range(self.pathmem.length())]
            for i in range(n):
                idx = np.random.choice(list(range(self.pathmem.length())), p=p)
                path = self.pathmem.get(idx)
                et = self.etmem.get(idx)[0]
                wgt = update_weights(cost_map, et, path, wgt)


        return wgt

    def moveTo(self, pos):
        #Tries to move to location. If wall in the way return false
        if environment[pos[0]][pos[1]] > 10:
            return False
        else:
            self.current_pos = pos
            return True

    def sense(self):
        #Gets information about the current locations through it's "Sensors" adds some gaussian noise to reading

        #Noise
        noise = np.random.normal(0, 0.25)

        return environment[self.current_pos[0]][self.current_pos[1]] + noise

    def drive(self, path, costmap, et):
        reversedpath = list(reversed(copy.deepcopy(path)))
        new_costmap = copy.deepcopy(costmap)

        #In eligibility trace memory, keep track of number of times it is accessed for replay
        self.etmem.push((et, 1))
        
        pathtemp = []

        for n, i in enumerate(reversedpath):

            completed = self.moveTo(i)
            if n == 0:
                if not completed:
                    print("Could not get to current destination (This shouldn't happen)")
                    print(i)
                    exit(1)
                continue

            if not completed:
                #If list is not empty
                if pathtemp:
                    self.pathmem.push(pathtemp)
                new_costmap[i[0]][i[1]] = 15
                return new_costmap, n

            self.expmap[self.current_pos[0]][self.current_pos[1]] *= 0.75
            pathtemp.append(i)
            new_costmap[self.current_pos[0]][self.current_pos[1]] = self.sense()

        #If list is not empty
        if pathtemp:
            self.pathmem.push(pathtemp)
        return new_costmap, len(path)