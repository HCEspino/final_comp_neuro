"""
  * **********************************************************************
  *
  * Copyright (c) 2014 Regents of the University of California. All rights reserved.
  *
  * Redistribution and use in source and binary forms, with or without
  * modification, are permitted provided that the following conditions
  * are met:
  *
  * 1. Redistributions of source code must retain the above copyright
  *    notice, this list of conditions and the following disclaimer.
  *
  * 2. Redistributions in binary form must reproduce the above copyright
  *    notice, this list of conditions and the following disclaimer in the
  *    documentation and/or other materials provided with the distribution.
  *
  * 3. The names of its contributors may not be used to endorse or promote
  *    products derived from this software without specific prior written
  *    permission.
  *
  * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
  * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
  * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
  * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR
  * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
  * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
  * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
  * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
  * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
  * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
  * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
  *
  * **********************************************************************
  % spikeWaveEprop - Calculates a path using a spiking neuron wave front
  %   algorithm. The cost of traversal is reflected in the axonal delays
  %   between neurons. Uses the e-prop algorithm from Bellec et al. to
  %   update the weights to reflect the map costs.
  %
  %   This version of the algorithm is designed to randomly explore a space using a Levy Flight distribution.
  %   It chooses waypoints and generates paths.  The paths are currently mapped to GPS coordinates for
  %   robot navigation.
  %
  %   The spikeWaveEprop is explained in detail of:
  %      Krichmar, J.L., Ketz, N.A., Pilly, P.K., and Soltoggio, A. (2021).
  %         Flexible Path Planning through Vicarious Trial and Error.
  %         bioRxiv, 2021.2009.2008.459317.
  %
  %   The e-prop algorithm was originally described in:
  %      Guillaume Bellec, Franz Scherr, Anand Subramoney, Elias Hajek,
  %         Darjan Salaj, Robert Legenstein, and Wolfgang Maass.
  %         A solution to the learning dilemma for recurrent networks of
  %         spiking neurons. Nature Communications, 11 (1):3625, 2020.
"""
import tarfile

from numpy import loadtxt
import numpy as np

"""
% delay_rule - Calculates a delta function for the weights. The weights hold
%   the axonal delay between neurons.
%
% @param wBefore - weight value before applying learning rule.
% @param value - value from the mapxy.
% @param learnRate - learning rate.
% @return - weight value after applying learning rule.
"""


def delay_rule(w_before, value, learn_rate):
    return learn_rate * (value - w_before)


"""
       Calculate the Euclidean distance 
       - Parameters
           x1, x2 - First and second element of the x vector 
           y1, y2 - First and second element of the y vector
       - Returns the distance between x and y
"""


def get_distance(x, y):
    return np.sqrt(pow(x[0] - y[0], 2.0) + pow(x[1] - y[1], 2.0))


"""
% get_path - Generates the path based on the AER spike table. The spike
%   table is ordered from earliest spike to latest. The algorithm starts at
%   the end of the table and finds the most recent spike from a neighbor.
%   This generates a near shortest path from start to end.
% 
% @param spks - AER  table containing the spike time and ID of each neuron.
% @param mapxy - mapxy of the environment.
% @param s - start location.
% @param e - end (goal) location.
% @return - path from start to end.
"""


def get_path(spks, mapxy, s, e):
    path = list([])
    path.append(e)

    # work from most recent to oldest in the spike table
    for i in range(len(spks) - 1, -1, -1):
        found = False
        lst = []
        pinx = len(path) - 1
        for j in range(len(spks)):
            # find the last spike from a neighboring neuron.
            if spks[j][0] == i and get_distance(path[pinx], [spks[j][1], spks[j][2]]) < 1.5:
                lst.append([spks[j][1], spks[j][2]])
                found = True

        # if there is more then one spike, find the one with the lowest cost and
        # closest to starting and/or end locations.
        minx = 0
        if found:
            if len(lst) > 1:
                cost = 999999999
                dist_start = 999999999
                dist_end = 999999999

                for m in range(len(lst)):
                    if mapxy[lst[m][0]][lst[m][1]] < cost:
                        cost = mapxy[lst[m][0]][lst[m][1]]
                        minx = m
                        dist_end = get_distance(e, lst[m])
                        dist_start = get_distance(s, lst[m])
                    elif mapxy[lst[m][0]][lst[m][1]] == cost and get_distance(s, lst[m]) < dist_start and get_distance(
                            e, lst[m]) < dist_end:
                        minx = m
                        dist_start = get_distance(s, lst[m])
                        dist_end = get_distance(e, lst[m])
                    elif mapxy[lst[m][0]][lst[m][1]] == cost and get_distance(s, lst[m]) < dist_start:
                        minx = m
                        dist_start = get_distance(s, lst[m])
                        dist_end = get_distance(e, lst[m])
                    elif mapxy[lst[m][0]][lst[m][1]] == cost and get_distance(e, lst[m]) < dist_end:
                        minx = m
                        dist_start = get_distance(s, lst[m])
                        dist_end = get_distance(e, lst[m])

            # add the neuron to the path. but don't add if it is the goal
            if get_distance(e, lst[minx]) > 0:
                path.append(lst[minx])

    return path


"""
 spike_wave - Calculates a path using a spiking neuron wave front
   algorithm. The cost of traversal is reflected in the axonal delays
   between neurons.

 @param weights - current weights.
 @param mapxy - grid mapxy. values in mapxy reflect cost of traversal.
 @param startx - x coordinate of starting location.
 @param starty - y coordinate of starting location.
 @param endx - x coordinate of goal location.
 @param endy - y coordinate of goal location.
 @return eligibility_trace - neuron eligibility trace.
 @return path - path generated by spike wave front.
"""


def spike_wave(weights, mapxy, startx, starty, endx, endy):
    n1 = mapxy.shape[0]
    n2 = mapxy.shape[1]
    spike = 1
    refractory = -5
    eligibility_trace_tc = 25

    # Each neuron connects to its 8 neighbors
    delay_buffer = np.zeros([n1, n2, n1, n2])
    v = np.zeros([n1, n2])  # Initial values of v voltage
    u = np.zeros([n1, n2])  # Initial values of u recovery
    eligibility_trace = np.zeros([n1, n2])  # Initial values of eligibility trace
    i_exc = np.zeros([n1, n2])

    # the spike wave is initiated from the starting location
    v[startx][starty] = spike

    found_goal = False
    time_steps = 0
    aer = []

    while not found_goal:
        time_steps = time_steps + 1

        # find the neurons that spiked this time step
        inx = 0
        fx = []
        fy = []
        for i in range(n1):
            for j in range(n2):
                if v[i][j] >= spike:
                    fx.append(i)
                    fy.append(j)
                    aer.append([time_steps, i, j])  # keep spike information in AER (spikeID and timeStep)
                    inx += 1

        # Neurons that spike send their spike to their post-synaptic targets.
        # The weights are updated and the spike goes in a delay buffer to
        # targets. The neuron's recovery variable is set to its refractory value.
        for i in range(inx):
            # print("%i: fx[%i]=%i fy[%i]=%i" % (time_steps, i, fx[i], i, fy[i]))
            u[fx[i]][fy[i]] = refractory
            eligibility_trace[fx[i]][fy[i]] = spike
            for j in range(n1):
                for k in range(n2):
                    if weights[fx[i]][fy[i]][j][k] > 0:
                        delay_buffer[fx[i]][fy[i]][j][k] = round(weights[fx[i]][fy[i]][j][k])
            if fx[i] == endx and fy[i] == endy:
                found_goal = True  # neuron at goal location spiked.

        # if the spike wave is still propagating, get the synaptic input for
        # all neurons. Synaptic input is based on recovery variable, and spikes
        # that are arriving to the neuron at this time step.
        if not found_goal:
            for i in range(n1):
                for j in range(n2):
                    i_exc[i][j] = u[i][j]

            for i in range(n1):
                for j in range(n2):
                    for k in range(n1):
                        for m in range(n2):
                            if delay_buffer[i][j][k][m] == 1 and weights[i][j][k][m] > 0:
                                i_exc[k][m] += 1
                            delay_buffer[i][j][k][m] = max(0, delay_buffer[i][j][k][m] - 1)
                    eligibility_trace[i][j] -= eligibility_trace[i][j] / eligibility_trace_tc

            # Update membrane potential (v) and recovery variable (u)
            for i in range(n1):
                for j in range(n2):
                    v[i][j] += i_exc[i][j]
                    u[i][j] = min(u[i][j] + 1, 0)
    path = get_path(aer, mapxy, [startx, starty], [endx, endy])  # Get the path from the AER table.

    return eligibility_trace, path

"""
 update_weights - Updates the weights according to the E-Prop learning rule.

 @param w - current weight values.
 @param mapxy - cost. values in cost reflect cost of traversal. Expecting an array of the same size as the map
 @param e_trace - eligibility trace of neurons.
 @param starty - y coordinate of starting location.
 @param path - path generated by spike wave front.
 @:return w - updated weights
"""


def update_weights(cost, e_trace, path, w):

    learning_rate = 0.1

    w1 = w.shape[0]
    w2 = w.shape[1]
    for i in range(len(path)):
        for j in range(w1):
            for k in range(w2):
                if w[path[i][0]][path[i][1]][j][k] > 0:
                    loss = cost[j][k] - w[path[i][0]][path[i][1]][j][k]  # Assume agent can sense surrounding region
                    w[path[i][0]][path[i][1]][j][k] += learning_rate * loss * e_trace[path[i][0]][path[i][1]]

    return w


"""
 get_gps_waypoints - Takes the path generated by the spike wave algorithm
   and returns the corresponding gps locations.

 @param path - path generated by spikewave front algorithm.
 @param lat - matrix of latitude coordinates corresponding to the cost map neurons.
 @param lon - matrix of longitude coordinates corresponding to the cost map neurons.
 @return gps - sequence of gps coordinates (latitude, longitude).
"""

"""
 levy_flight - Gets a waypoint to explore using the Levy Flight distribution.
   
   Taken from Section 3.3 of: 
       Yang, X.S. (2014). Random Walks and Optimization. Nature-Inspired Optimization Algorithms, Elsevier Inc. 45-65.
       
 @param dims - number of dimensions. 2 would correspond to coordinates on a map.
 @param step_size - roughly the average step to move.
 @return lf - an array of size dims that would be the displacement from current position.
"""


def levy_flight(dims, step_size):
    step = []
    lf = []

    beta = 1.5
    gamma1 = 1.3293  # equals the gamma function with beta + 1
    gamma2 = 0.9064  # equals the gamma function with (beta+1)/2
    sigma = (gamma1*np.sin(np.pi * beta/2) / (gamma2*beta*2**((beta - 1) / 2)))**(1 / beta)
    u = np.random.normal(0, sigma**2, dims)
    v = np.random.normal(0, 1, dims)
    for i in range(dims):
        step.append(u[i]/abs(v[i])**(1/beta))
        lf.append(step_size*step[i])

    return lf


"""
Main Routine
   Load a map
   Call spike_wave
   spike_wave returns the path
"""