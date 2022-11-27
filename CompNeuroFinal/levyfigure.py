#Run this to get the figure of the levy flight distribution.
import matplotlib.pyplot as plt
from spikeWaveEprop import levy_flight

xs = [1.0]
ys = [1.0]

for i in range(10):
    lf = levy_flight(2, 3)  # levy_flight will return a new waypoint
    # waypoint coordinates must be whole numbers

    #Keep going until positive results
    while lf[0] < 0 or lf[0] > 10 or lf[1] < 0 or lf[1] > 10:
        lf = levy_flight(2, 3)

    xs.append(lf[0])
    ys.append(lf[1])

plt.plot(xs, ys, 'bo', linestyle='--')
plt.title("Levy flight exploration for 10 trials")
plt.show()

dist = []
for i in range(1000):
    lf = levy_flight(1, 3)
    #Keep going until positive results
    while lf[0] < 0 or lf[0] > 10:
        lf = levy_flight(1, 3)
    dist.append(lf[0])


plt.hist(dist)
plt.title("Levy flight distribution, 1000 samples")
plt.show()