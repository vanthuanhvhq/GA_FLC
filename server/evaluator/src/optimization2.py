#!/usr/bin/env python
# coding: utf-8

# # Optimization of Fuzzy Logic Controller Used for a Differential Drive Wheeled Mobile Robot

# In[ ]:


_runDemos = True # this is queried in this document later multiple times


# ## Cluster Query

# This task could use for its solution a computer cluster. As it is optional, use it only when you are running it. In this case you must set the option ```useCluster``` to ```True``` and set proper value in ```url``` variable.

# In[ ]:


useCluster = False
url = 'https://ourserver/api/evaluator/FFFF'

if useCluster:
    url = input('Enter full / https include / cluster URL: \t')


# In[ ]:


# get_ipython().system('pip install scikit-fuzzy')


# In[ ]:


import numpy as np
import math
import random
from math import *
import scipy.integrate as integrate
import matplotlib.pyplot as plt
import pandas as pd
# get_ipython().run_line_magic('matplotlib', 'inline')
import skfuzzy as fuzz


# ## Robot parameters

# In[ ]:


robotState0 = {
    'x': 0,
    'y': 0,
    'theta': -3.14 / 4
}

robotParams = {
    'r': 0.0925,
    'b': 0.37,
    'm': 9,
    'I': 0.16245,
}


# ## Robot models

# ### Kinematic model

# In[ ]:


from math import sin, cos
def createRobot(params):
    m = params['m']
    I = params['I']

    def robot(t, currentState, controller):
        # ask controller for velocity and omega
        velocity, omega = controller(t, currentState)

        currentTheta = currentState[2]
        cosTheta = cos(currentTheta)
        sinTheta = sin(currentTheta)

        x_dot = velocity * cosTheta
        y_dot = velocity * sinTheta
        theta_dot = omega

        E = 0.5 * (m * velocity * velocity + I * omega * omega)

        return [x_dot, y_dot, theta_dot, velocity, omega, E] # velocity, omega, E are returned for easy evaluation they are not needed for computation
    return robot

robot = createRobot(robotParams)


# ### Dynamic Model

# Dynamic model extends kinematic model with differential equations describing the motors.

# In[ ]:


# example of motor parameters,
motorParams = {
    'J': 0.01,
    'B': 0.1,
    
    'Kt': 0.01,
    'Ke': 0.01,
    'K': 0.01,
    
    'Ra': 0.1,
    'La': 0.01
}

#//////////////////////////////////////////////////////////////////////////////
def createFilter2ndOrder(b1, b0, a1, a0):
    def filter2ndOrder(t, u, currentState):
        x0 = currentState[0]
        x1 = currentState[1]
        dx0 = b0 * u + a0 * x0 + x1
        dx1 = b1 * u + a1 * x0
        return [dx0, dx1]
    return filter2ndOrder

def createMotorModel(motorParams=None):
    if motorParams is None:
        return None
    
    K = motorParams['K']
    J = motorParams['J']
    La = motorParams['La']
    Ra = motorParams['Ra']
    B = motorParams['B']
    
    b1 = K / (La * J)
    b0 = 0
    a1 = -(Ra * B + K * K) / (La * J)
    a0 = -(Ra * J + La * B) / (La * J)
    return createFilter2ndOrder(b1, b0, a1, a0)
    

def createRobotModelWithDynamic(params, motorModel = None):
    """
    function returns standard ODE model usable in many libraries (scipy)
    """
    # motorAsFilter = createFilter2ndOrder(b1, b0, a1, a0)
    m = params['m']
    I = params['I']
    b = params['b']

    motorAsFilter = motorModel
    def robotWithDynamic(t, currentState, controller):
        # ask controller for velocity and omega
        velocity, omega = controller(t, currentState)

        delta = omega * b / 2
        vL = velocity - delta
        vR = velocity + delta
        vLState = currentState[6:8]
        vRState = currentState[8:10]
        vLStateD = motorAsFilter(t, vL, vLState)
        vRStateD = motorAsFilter(t, vR, vRState)
        vLFiltered = vLState[0]
        vRFiltered = vRState[0]

        velocity = (vRFiltered + vLFiltered) / 2
        delta = (vRFiltered - vLFiltered) / 2
        omega = 2 * delta / b

        currentTheta = currentState[2]
        cosTheta = cos(currentTheta)
        sinTheta = sin(currentTheta)

        x_dot = velocity * cosTheta
        y_dot = velocity * sinTheta
        theta_dot = omega

        E = 0.5*(m*(velocity)*(velocity) + I*(omega)*(omega))

        return [x_dot, y_dot, theta_dot, velocity, omega, E, *vLStateD, *vRStateD] #velocity, omega, E are returned for easy evaluation they are not needed for computation

    def robot(t, currentState, controller):
        """
        This closure is result of parent function
        """
        # ask controller for velocity and omega
        velocity, omega = controller(t, currentState)

        currentTheta = currentState[2]
        cosTheta = cos(currentTheta)
        sinTheta = sin(currentTheta)

        x_dot = velocity * cosTheta
        y_dot = velocity * sinTheta
        theta_dot = omega

        E = 0.5 * (m * velocity * velocity + I * omega * omega)

        return [x_dot, y_dot, theta_dot, velocity, omega, E] #velocity, omega, E are returned for easy evaluation they are not needed for computation

    if motorModel is None:
        return robot
    else:
        return robotWithDynamic
    pass
  
motorModel = createMotorModel(motorParams)    
robotWithDynamic = createRobotModelWithDynamic(robotParams, motorModel)
robot = robotWithDynamic # If you delete / comment this line, only the Kinematic model is taken into account.


# ## Solver

# In[ ]:


# selectors are defined for extration of data from results computed by ODE solver
selectx = lambda item: item['y'][0]       # x position
selecty = lambda item: item['y'][1]       # y position
selectt = lambda item: item['time']       # time
selectv = lambda item: item['dy'][3]      # velocity
selectomega = lambda item: item['dy'][2]  # omega = theta_dot
selecte = lambda item: item['TotalEnergy']# total energy
selects = lambda item: item['y'][3]       # displacement
selectors = {
    'time': selectt,
    'x': selectx, 
    'y': selecty, 
    'd': selects, 
    'v': selectv, 
    'omega': selectomega,
    'E': selecte}

#          yIndex=0, yIndex=1, yIndex=2, yIndex=3, yIndex=4, yIndex=5, yIndex=6,
def compute(model, state0, t0 = 0.0, t_bound = 10, max_step = 0.05):
    """
    This function returns a generator containing the sequence of resuls. 
    In this particular case it will return a sequence of robot states.
    """
    solver = integrate.RK45(fun = model, t0 = t0, y0 = state0, t_bound = t_bound, max_step = max_step)
    cnt = 0
    lastEnergy = 0
    totalEnergy = 0

    #names = ['t', 'x', 'y', 'θ', 's', 'θ2', 'IE', "x'", "y'", 'ω', 'v', 'ω2', 'E']
    while True:
        message = solver.step()
        #currentItem = [solver.t, *solver.y, *model(solver.t, solver.y)]
        currentItem = {'time': solver.t, 'y': solver.y, 'dy': model(solver.t, solver.y)}
        #t, 'solver.y': x, y, theta, s, theta, intE 'model': x', y', theta', velocity, omega, E
        #0,             0, 1,   2,   3,   4,     5,          0,  1,   2,        3,      4,    5  
        # Energy calculation / energy sumation
        currentEnergy = currentItem['dy'][5] #currentNamed['E']
        deltaEnergy = currentEnergy - lastEnergy

        if deltaEnergy > 0:
            totalEnergy = totalEnergy + deltaEnergy

        lastEnergy = currentEnergy
        currentItem['TotalEnergy'] = totalEnergy

        yield currentItem
        if (not(solver.status == 'running')):
            break
    return


# ## Path Controller

# Path Controller transforms a controller navigating robot towards a fixed distance to controller able switch the destination immediately after reaching point defined by a path.

# In[ ]:


def controllerForPath(controller, path, distanceEps=1e-2):
    destinationX, destinationY, destinationOrietation = next(path)
    destinationState = [destinationX, destinationY, destinationOrietation]
    lastReached = False
    #print('Destination set to', destinationState)
    def result(t, currentState):
        """
        This closure is result of parent function and acts as a controller - mediator,
        which commands the given controller.
        """
        nonlocal destinationX # use parent variable
        nonlocal destinationY # use parent variable
        nonlocal destinationState # use parent variable
        nonlocal lastReached # use parent variable

        currentX = currentState[0]
        currentY = currentState[1]
        deltaX = destinationX - currentX
        deltaY = destinationY - currentY
        if (lastReached == False):
          # last point in path was not reached
          if (deltaX * deltaX + deltaY * deltaY < distanceEps):
            # robot is close enought to currentDestination
            try:
                # try to get another point on path
                destinationX, destinationY, destinationOrietation = next(path)
                destinationState = [destinationX, destinationY, destinationOrietation]
                #print('Destination set to', destinationState, 'while in state', currentState)
            except StopIteration:
              # there are no other points
              lastReached = True
        if (lastReached):
            return (0, 0)
        else:
            return controller(t, currentState, destinationState)
    return result


# ## Model Creator

# Model creator is function which packs all subsystems into one described by standard ODE function. Standard methods for ODE problems could be applied on such result / function.

# In[ ]:


def robotModelCreator(controllerCreator, path, **kwargs):
    controller_ = controllerCreator(**kwargs)
    savedController = controllerForPath(controller_, path)
    def resultRMC(t, currentState):
        return robot(t, currentState, savedController)
    return resultRMC


# ## Computation

# Simple compute allows to fully define parameters at first and then use it on model. Such approach is usefull in case when different models (controllers) are used for same task. In this case this function simplify comparison of different controllers.

# In[ ]:


def simpleCompute(computefunc, state0, t0 = 0, t_bound = 200, max_step = 0.05):
    def resultSC(model):
        return computefunc(
          model, state0 = state0, t0 = t0, t_bound = t_bound, max_step = max_step)
    return resultSC


# ## Controllers

# All controllers have to have same signature (parameter list)
# 
# ```python
# def controller(t, currentState, destinationState)
# ```
# 
# thus a creator taking special controller parameters must be defined. Such a creator should accept all special parameters and return controller with standard signature.

# ### Circle Controller

# In[ ]:


def createCircleControllerWithGain(gain, omega_ri, vri, lowVelocityLimit, highVelocityLimit, lowOmegaLimit, highOmegaLimit):
    def controller(t, currentState, destinationState):
        currentX = currentState[0]
        currentY = currentState[1]
        currentTheta = currentState[2]

        destinationX = destinationState[0]
        destinationY = destinationState[1]

        cosTheta = cos(currentTheta)
        sinTheta = sin(currentTheta)

        deltaX = destinationX - currentX
        deltaY = destinationY - currentY

        velocity = vri
        omega = -2 * gain * vri * (deltaX * sinTheta - deltaY * cosTheta) / (deltaX * deltaX + deltaY * deltaY)
    
        if (velocity > highVelocityLimit):
            velocity = highVelocityLimit
        if (velocity < lowVelocityLimit):
            velocity = lowVelocityLimit
        if (omega > highOmegaLimit):
            omega = highOmegaLimit
        if (omega < lowOmegaLimit):
            omega = lowOmegaLimit

        return velocity, omega
    return controller


# #### Full Example of Use

# In[ ]:


def localDemo():
    pathForSimulation = iter([
            [0, 0, 0],  #X, Y, orientation
            [10, 0, 0], #X, Y, orientation
            [10, 10, 0], #X, Y, orientation
            [0, 10, 0], #X, Y, orientation
            [0, 0, 0]
        ])

    pathForSimulation = iter([
            [0, 0, 0],  #X, Y, orientation
            [10, 0, 0], #X, Y, orientation
            [10, 10, 0], #X, Y, orientation
            [20, 10, 0], #X, Y, orientation
            [20, 20, 0]
        ])


    robotState0 = {
            'x': 0,
            'y': 0,
            'theta': -3.14 / 4
        }

    t0 = 0
    t_bound = 100
    max_step = 0.05

    state0 = None
    if robot == robotWithDynamic:
        state0 = np.array([robotState0['x'], robotState0['y'], robotState0['theta'], 0, 0, 0, 0, 0, 0, 0]) # x0=0, y0=0, theta
    else:
        state0 = np.array([robotState0['x'], robotState0['y'], robotState0['theta'], 0, 0, 0]) # x0=0, y0=0,theta

    solverfunc = simpleCompute(
        compute, state0 = state0, 
        t0 = t0, t_bound = t_bound, max_step = max_step)    

    controllerParams = {
        'gain': 4, 
        'omega_ri': 0, 
        'vri': 2.0, 
        'lowVelocityLimit': 0.2, 
        'highVelocityLimit': 2.0, 
        'lowOmegaLimit': -0.75, 
        'highOmegaLimit': 0.75
        }

    fullRobot = robotModelCreator(createCircleControllerWithGain, pathForSimulation, **controllerParams)      
    state1 = fullRobot(0, state0)
    robotStates = solverfunc(fullRobot)

    results = {}
    for key, selector in selectors.items():
        print(key)
        results[key] = []

    for currentState in robotStates: # readout all states from current moving robot
        for key, selector in selectors.items():
            results[key].append(selector(currentState))

    plt.plot(results['x'], results['y'])

if _runDemos:
    localDemo()


# ### Robins

# In[ ]:


def createController_By_RobinsMathew(k0, k1, omega_ri, vri, lowVelocityLimit, highVelocityLimit, lowOmegaLimit, highOmegaLimit):
    def controller(t, currentState, destinationState):
        currentX = currentState[0]
        currentY = currentState[1]
        currentTheta = currentState[2]

        destinationX = destinationState[0]
        destinationY = destinationState[1]

        cosTheta = cos(currentTheta)
        sinTheta = sin(currentTheta)
    
        deltaX = destinationX - currentX
        deltaY = destinationY - currentY
        theta_destination = atan2(deltaY, deltaX)
        theta_error = theta_destination - currentTheta

        Te = math.sin(theta_destination)*deltaX - math.cos(theta_destination)*deltaY
    
        velocity = vri*math.cos(theta_error)
        omega = omega_ri + k0*vri*Te + k1*vri*math.sin(theta_error)

        if velocity > highVelocityLimit:
            velocity = highVelocityLimit
        if (velocity < lowVelocityLimit):
            velocity = lowVelocityLimit
        if omega > highOmegaLimit:
            omega = highOmegaLimit
        if (omega < lowOmegaLimit):
            omega = lowOmegaLimit
      
        return velocity, omega
    return controller


# ### Fuzzy Logic Controller

# #### Helper Functions

# In[ ]:


def createFuzzyfier(space, categories, trimf = fuzz.trimf, membership = fuzz.interp_membership):
    fuzzyInput = {}
    for key, value in categories.items():
        fuzzyInput[key] = trimf(space, value)
    def result(variable):
        output = {}
        for key, value in fuzzyInput.items():
            output[key] = membership(space, value, variable)
        if output[key] ==0:
            output[key] = 1e-5
        else:
            output[key] = output[key] 
        return output
    return result

def createInferenceSystem(inputAfuzzyfier, inputBfuzzyfier, outputSpace, outputDict, rulesDict, trimf = fuzz.trimf):
    fuzzyResults = {}
    for keyA, outerValue in rulesDict.items():
        if not(keyA in fuzzyResults):
            fuzzyResults[keyA] = {}
        for keyB, innerValue in outerValue.items():
            fuzzyResults[keyA][keyB] = trimf(outputSpace, outputDict[innerValue]) #innerValue==outputDict[keyA][keyB]
    def result(valueA, valueB):
        fuzzyVariableA = inputAfuzzyfier(valueA)
        fuzzyVariableB = inputBfuzzyfier(valueB)
        fuzzyResult = None
        for keyA, outerValue in rulesDict.items():
            for keyB, resultValue in outerValue.items():
                currentResult = np.fmin(fuzzyResults[keyA][keyB],
                    np.fmin(fuzzyVariableA[keyA], fuzzyVariableB[keyB]))
                if fuzzyResult is None:
                    fuzzyResult = currentResult
                else:
                    fuzzyResult = np.fmax(currentResult, fuzzyResult)
        return fuzzyResult
    return result

def createDefuzzyfier(outputSpace, *defuzzArgs, defuzz=fuzz.defuzz, **defuzzKwargs):
    def result(value):
        return defuzz(outputSpace, value, *defuzzArgs, **defuzzKwargs)
    return result
  
def createFullFuzzySystem(inferenceSystem, defuzzyfier):
    def system(inputA, inputB):
        return defuzzyfier(inferenceSystem(inputA, inputB))
    return system


# #### Controller

# In[ ]:


def createFuzzyController(fuzzyDescription, r, b, omega_ri, vri, lowVelocityLimit, highVelocityLimit, lowOmegaLimit, highOmegaLimit):
    inputsDistance = fuzzyDescription['inputs']['distance']['M']
    inputsSpaceDistance = np.array(fuzzyDescription['inputs']['distance']['S'])
    
    inputsAngle = fuzzyDescription['inputs']['angle']['M']
    inputsSpaceAngle = np.array(fuzzyDescription['inputs']['angle']['S'])
    
    outputsOmegaR = fuzzyDescription['outputs']['omegaR']['M']
    outputSpaceOmegaR = np.array(fuzzyDescription['outputs']['omegaR']['S'])
    outputRulesOmegaR = fuzzyDescription['outputs']['omegaR']['rules']
    
    outputsOmegaL = fuzzyDescription['outputs']['omegaL']['M']
    outputSpaceOmegaL = np.array(fuzzyDescription['outputs']['omegaL']['S'])
    outputRulesOmegaL = fuzzyDescription['outputs']['omegaL']['rules']


    inputsDistanceFuzzyfier = createFuzzyfier(inputsSpaceDistance, inputsDistance)
    inputsAngleFuzzyfier = createFuzzyfier(inputsSpaceAngle, inputsAngle)

    inferenceSystem_R = createInferenceSystem(inputsDistanceFuzzyfier, inputsAngleFuzzyfier, outputSpaceOmegaR, outputsOmegaR, outputRulesOmegaR)
    outputDefuzzyfier_R = createDefuzzyfier(outputSpaceOmegaL, mode='centroid')

    inferenceSystem_L = createInferenceSystem(inputsDistanceFuzzyfier, inputsAngleFuzzyfier, outputSpaceOmegaL, outputsOmegaL, outputRulesOmegaL)
    outputDefuzzyfier_L = createDefuzzyfier(outputSpaceOmegaL, mode='centroid')

    fullSystem_R = createFullFuzzySystem(inferenceSystem_R, outputDefuzzyfier_R)
    fullSystem_L = createFullFuzzySystem(inferenceSystem_L, outputDefuzzyfier_L)
    
    def controller(t, currentState, destinationState):
        currentX = currentState[0]
        currentY = currentState[1]
        currentTheta = currentState[2]

        destinationX = destinationState[0]
        destinationY = destinationState[1]

        cosTheta = cos(currentTheta)
        sinTheta = sin(currentTheta)
        
        deltaX = destinationX - currentX
        deltaY = destinationY - currentY
        theta_destination = atan2(deltaY, deltaX)
        THETA_ERROR = theta_destination - currentTheta
        DISTANCE_ERROR = sqrt(deltaX * deltaX + deltaY * deltaY)
        
        if (THETA_ERROR > pi):
            THETA_ERROR -= 2*pi
        if (THETA_ERROR < -pi):
            THETA_ERROR += 2*pi
      
        omega_R = fullSystem_R(DISTANCE_ERROR, THETA_ERROR)
        omega_L = fullSystem_L(DISTANCE_ERROR, THETA_ERROR)

        velocity = r * (omega_R + omega_L) / 2
        omega = r * (omega_R - omega_L) / b

        if velocity > highVelocityLimit:
            velocity = highVelocityLimit
        if (velocity < lowVelocityLimit):
            velocity = lowVelocityLimit
        if omega > highOmegaLimit:
            omega = highOmegaLimit
        if (omega < lowOmegaLimit):
            omega = lowOmegaLimit

        return velocity, omega
    return controller


# ## Simulation Function

# In next part the full description of simulation is stored in a single structured JSON document / variable. If this document is mutated, the slighly different condition for simulation are defined. Set of mutated documents and results of described simulations might be compared and thus proper results can be selected. This process creates a basement for optimization techniques.

# ### Simulation Description

# In[ ]:


simulationDescription = {

    'robotState0': {
        'x': 0,
        'y': 0,
        'theta': -3.14 / 4
    },

    'path': [
        [0, 0, 0],  #X, Y, orientation
        [10, 0, 0], #X, Y, orientation
        [10, 10, 0], #X, Y, orientation
        [20, 10, 0], #X, Y, orientation
        [20, 20, 0]
    ],

    'robotParams': {
        'r': 0.0925,
        'b': 0.37,
        'm': 9,
        'I': 0.16245,
        #'motorParams': None,
        'motorParams': {
            'J': 0.01,
            'B': 0.1,

            'Kt': 0.01,
            'Ke': 0.01,
            'K': 0.01,

            'Ra': 0.1,
            'La': 0.01
        }
    },
    
    'controllerParams': {
        'omega_ri': 0, 'vri': 2.0,'lowVelocityLimit': 0.2, 
        'highVelocityLimit': 2.0, 'lowOmegaLimit': -0.75, 'highOmegaLimit': 0.75
    },

    'simulationParams': {
        't0': 0,
        't_bound': 100,
        'max_step': 0.05
    }
}


# ### Executor

# In[ ]:


def runSimulation(simulationDescription, controllerCreator, selectors=selectors):
  
    pathForSimulation = iter(simulationDescription['path'])

    t0 = simulationDescription['simulationParams']['t0']
    t_bound = simulationDescription['simulationParams']['t_bound']
    max_step = simulationDescription['simulationParams']['max_step']

    state0 = None
    robotState0 = simulationDescription['robotState0']
    if robot == robotWithDynamic:
        state0 = np.array([robotState0['x'], robotState0['y'], robotState0['theta'], 0, 0, 0, 0, 0, 0, 0]) # x0=0, y0=0, theta
    else:
        state0 = np.array([robotState0['x'], robotState0['y'], robotState0['theta'], 0, 0, 0]) # x0=0, y0=0,theta

    solverfunc = simpleCompute(
        compute, state0 = state0, 
        t0 = t0, t_bound = t_bound, max_step = max_step)

    controllerParams = simulationDescription['controllerParams']
    completeRobot = robotModelCreator(controllerCreator, pathForSimulation, **controllerParams)      
    robotStates = solverfunc(completeRobot)

    results = {}
    for key, selector in selectors.items():
        results[key] = []

    for currentState in robotStates: # readout all states from current moving robot
        for key, selector in selectors.items():
            results[key].append(selector(currentState))

    return results


# ### Example of Use

# In[ ]:


import copy

def localDemo():
    circleControllerDescription = copy.deepcopy(simulationDescription)
    circleControllerDescription['controllerParams']['gain'] = 4

    results = runSimulation(circleControllerDescription, createCircleControllerWithGain, selectors)
    plt.plot(results['x'], results['y'])
    
if _runDemos:
    localDemo()


# ## Chromozome Mapping Functions

# These functions change standard simulation description into description based on information stored in chromosome. Also these functions could be named as a chromosome information decoder.

# ### Circle Controller

# In[ ]:


import copy
def fromChromozomeToDescriptionCircle(chromozome, description):
    result = copy.deepcopy(description)
    result['controllerParams']['gain'] = chromozome[0]
    return result


# ### Robins Controller

# In[ ]:


import copy
def fromChromozomeToDescriptionRobins(chromozome, description):
    result = copy.deepcopy(description)
    result['controllerParams']['k0'] = chromozome[0]
    result['controllerParams']['k1'] = chromozome[1]
    return result


# ### Fuzzy Logic Controller

# In[ ]:


import copy
def fromChromozomeToDescriptionFuzzy(chromozome, description):
    CH = chromozome # just for simplicity
    result = copy.deepcopy(description)

    fuzzyDescription = {
        'inputs': {
            'distance' : {
                'S': list(np.arange(0, 2, 0.02)),
                'M': {'VC': [0, 0, 0.5], 'C': [0, 0.5, 1], 'M': [0.5, 1, 1.5], 'F': [1, 1.5, 2], 'VF': [1.5, 2, 2]}
            },
            'angle' : {
                'S': list(np.arange(-3.14, 3.14, 0.0628)),
                'M': {'BN': [-3.14, -3.14, -1.57], 'N': [-3.14, -1.57, 0], 'Z': [-1.57, 0, 1.57], 'P': [0, 1.57, 3.14], 'BP': [1.57, 3.14, 3.14]}
            }
        },
        'outputs': {
            'omegaR': {
                'S': list(np.arange(0, 30, 0.3)),
                'rules': {
                    'VC': {'BN': 'VSR', 'N': 'SR', 'Z': 'VSR', 'P': 'BR', 'BP': 'VBR'},
                    'C': {'BN': 'VSR', 'N': 'SR', 'Z': 'SR', 'P': 'BR', 'BP': 'VBR'},
                    'M': {'BN': 'VSR', 'N': 'SR', 'Z': 'MBR', 'P': 'BR', 'BP': 'VBR'},
                    'F': {'BN': 'VSR', 'N': 'SR', 'Z': 'BR', 'P': 'BR', 'BP': 'VBR'},
                    'VF': {'BN': 'VSR', 'N': 'SR', 'Z': 'VBR', 'P': 'BR', 'BP': 'VBR'}
                },
                'mode': 'centroid',
                'M': {'VSR': [0, 0, 7.5], 'SR': [0, 7.5, 15], 'MBR': [7.5, 15, 22.5], 'BR': [15, 22.5, 30], 'VBR': [22.5, 30, 30]}
            },
            'omegaL': {
                'S': list(np.arange(0, 30, 0.3)),
                'rules': {
                    'VC': {'BN': 'VBL', 'N': 'BL', 'Z': 'VSL', 'P': 'SL', 'BP': 'VSL'},
                    'C': {'BN': 'VBL', 'N': 'BL', 'Z': 'SL', 'P': 'SL', 'BP': 'VSL'},
                    'M': {'BN': 'VBL', 'N': 'BL', 'Z': 'MBL', 'P': 'SL', 'BP': 'VSL'},
                    'F': {'BN': 'VBL', 'N': 'BL', 'Z': 'BL', 'P': 'SL', 'BP': 'VSL'},
                    'VF': {'BN': 'VBL', 'N': 'BL', 'Z': 'VBL', 'P': 'SL', 'BP': 'VSL'} 
                },
                'mode': 'centroid',
                'M': {'VSL': [0, 0, 7.5], 'SL': [0, 7.5, 15], 'MBL': [7.5, 15, 22.5], 'BL': [15, 22.5, 30], 'VBL': [22.5, 30, 30]}
            }
        }
    }     

    distance_Member = {'VC': [0, 0, CH[0]], 
                       'C': [CH[2] - CH[1], CH[2], CH[2] + CH[3]],
                       'M': [CH[5] - CH[4], CH[5], CH[5] + CH[6]],
                       'F': [CH[8] - CH[7], CH[8], CH[8] + CH[9]], 
                       'VF': [2 - CH[10], 2, 2]}
    fuzzyDescription['inputs']['distance']['M'] = distance_Member

    angle_Member = {'BN': [-3.14, -3.14, -3.14+CH[11]], 
                    'N': [CH[13] - CH[12], CH[13], CH[13] + CH[14]],
                    'Z': [CH[16] - CH[15], CH[16], CH[16] + CH[17]], 
                    'P': [CH[19] - CH[18], CH[19], CH[19] + CH[20]], 
                    'BP': [3.14 - CH[21], 3.14, 3.14]}    
    fuzzyDescription['inputs']['angle']['M'] = angle_Member

    omegaR_Member =  {'VSR': [0, 0, CH[22]], 
                      'SR': [CH[24] - CH[23], CH[24], CH[24] + CH[25]],
                      'MBR': [CH[27] - CH[26], CH[27], CH[27] + CH[28]], 
                      'BR': [CH[30] - CH[29], CH[30], CH[30] + CH[31]], 
                      'VBR': [30 - CH[32], 30, 30]}
    fuzzyDescription['outputs']['omegaR']['M'] = omegaR_Member

    omegaL_Member =  {'VSL': [0, 0, CH[33]], 
                      'SL': [CH[35] - CH[34], CH[35], CH[35] + CH[36]],
                      'MBL': [CH[38] - CH[37], CH[38], CH[38] + CH[39]], 
                      'BL': [CH[41] - CH[40], CH[41], CH[41] + CH[42]], 
                      'VBL': [30 - CH[43], 30, 30]}
    fuzzyDescription['outputs']['omegaL']['M'] = omegaL_Member

    result['controllerParams']['fuzzyDescription'] = fuzzyDescription
    
    result['controllerParams']['r'] = result['robotParams']['r']
    result['controllerParams']['b'] = result['robotParams']['b']
    return result


# #### Demo of Use

# In[ ]:


def localDemo():
    _chromosome = [ 6.25366018e-01,  1.46639889e+00,  6.86498991e-02,  7.68093084e-01,
  2.94696025e-01,  8.19657937e-01,  7.66707859e-01,  3.30363761e-01,
  1.59154795e+00,  3.82371835e-01,  3.19164222e-01,  2.25863186e-01,
  5.40214083e+00, -2.87811455e+00,  3.22315195e+00,  5.24697655e+00,
 -2.18191386e-02,  2.35114508e-02,  2.29313174e+00,  2.71500790e+00,
  1.72456119e+00,  3.34836959e+00,  1.62090031e+01,  2.21135246e+01,
  4.62340233e+00,  9.07996460e+00,  2.99889795e+01,  1.59696212e+01,
  1.20020626e+01,  2.32626634e+01,  1.81445098e+01,  1.61965141e+01,
  2.16820148e+01,  5.66692201e+00,  7.02338927e-01,  2.90220179e+00,
  1.46153727e+01,  1.93699985e+01,  1.47587643e+01,  6.76618158e+00,
  4.15791676e+00,  2.87105164e+01,  1.39444156e+01,  1.14641525e+01]

    fuzzyLogicSimulationDescription = fromChromozomeToDescriptionFuzzy(_chromosome, simulationDescription)
    results = runSimulation(fuzzyLogicSimulationDescription, createFuzzyController, selectors=selectors)

    plt.plot(results['x'], results['y'])
    
if _runDemos:
    localDemo()


# ## Path Mapping Function

# This function allows to easy change simulation description for path which a robot has to follow.

# In[ ]:


import copy
def fromPathToDescription(path, description):
    result = copy.deepcopy(description)
    result['path'] = list(path)
    return result


# ## Fitness Functions (Based on Chromozomes)

# For optimization the fitness functions are needed. The simulation function has been defined earlier in this document. A fitness function must transform chromozome into simulation description, run appropriate simulation and return all results or subset of results. This chapter defines a bunch of fitness functions.

# ### Helper Functions

# #### Function for Fitness Function Creation

# ```createFitnessFunction``` allows to create fitness function which is appropriate for description, chromosome decoder (```mapperFunction```), function which creates controller (```controllerCreator```) and, if needed, selection of subresult (```resultSelector```).

# In[ ]:


def createFitnessFunction(baseDescription, mapperFunction, controllerCreator, resultSelector=lambda item: item):
    def fitnessFunction(chromozome):
        freshDescription = mapperFunction(chromozome, baseDescription)
        results = runSimulation(freshDescription, controllerCreator, selectors=selectors)
        result = resultSelector(results)
        return result
    return fitnessFunction


# In[ ]:


def energySelector(results):
    return results['E'][-1]


# In[ ]:


def distanceSelector(results):
    return results['d'][-1]


# #### Multivalue Functions

# ```singleAsMultiValue``` transforms function with scalar value into function with vector value. Such transformation creates a functionc which can evaluate multiple values in single call.

# In[ ]:


def singleAsMultiValue(singleFunction):
    def resultFunction(chromosomes):
        results = []
        for chromosome in chromosomes:
            results.append(singleFunction(chromosome))
        return results
    return resultFunction


# ### Simulation Description

# From this point simulations depend on ```simulationDescription``` defined in next code. Thus if any change is needed this is best place for it.

# In[ ]:


simulationDescription = {
    'robotState0': {
        'x': 0,
        'y': 0,
        'theta': -3.14 / 4
    },

    'path': [
        [0, 0, 0],  #X, Y, orientation
        [10, 0, 0], #X, Y, orientation
        [10, 10, 0], #X, Y, orientation
        [20, 10, 0], #X, Y, orientation
        [20, 20, 0]
    ],

    'robotParams': {
        'r': 0.0925,
        'b': 0.37,
        'm': 9,
        'I': 0.16245,
        #'motorParams': None,
        'motorParams': {
            'J': 0.01,
            'B': 0.1,

            'Kt': 0.01,
            'Ke': 0.01,
            'K': 0.01,

            'Ra': 0.1,
            'La': 0.01
        }
    },
    
    'controllerParams': {
        'omega_ri': 0, 'vri': 2.0,'lowVelocityLimit': 0.2, 
        'highVelocityLimit': 2.0, 'lowOmegaLimit': -0.75, 'highOmegaLimit': 0.75
    },

    'simulationParams': {
        't0': 0,
        't_bound': 100,
        'max_step': 0.05
    }
}


# ### Fuzzy Logic Controller

# #### Singlevalue Functions

# In[ ]:


fitnessFunctionFLC_Energy = createFitnessFunction(simulationDescription, fromChromozomeToDescriptionFuzzy, createFuzzyController, energySelector)
fitnessFunctionFLC_Distance = createFitnessFunction(simulationDescription, fromChromozomeToDescriptionFuzzy, createFuzzyController, distanceSelector)
fitnessFunctionFLC_FullResults = createFitnessFunction(simulationDescription, fromChromozomeToDescriptionFuzzy, createFuzzyController)


# #### Multivalue Functions

# In[ ]:


fitnessFunctionFLC_EnergyMulti = singleAsMultiValue(fitnessFunctionFLC_Energy)
fitnessFunctionFLC_DistanceMulti = singleAsMultiValue(fitnessFunctionFLC_Distance)
fitnessFunctionFLC_FullResultsMulti = singleAsMultiValue(fitnessFunctionFLC_FullResults)


# ### Circle

# #### Singlevalue Functions

# In[ ]:


fitnessFunctionCircle_Energy = createFitnessFunction(simulationDescription, fromChromozomeToDescriptionCircle, createCircleControllerWithGain, energySelector)
fitnessFunctionCircle_Distance = createFitnessFunction(simulationDescription, fromChromozomeToDescriptionCircle, createCircleControllerWithGain, distanceSelector)
fitnessFunctionCircle_FullResults = createFitnessFunction(simulationDescription, fromChromozomeToDescriptionCircle, createCircleControllerWithGain)


# #### Multivalue Functions

# In[ ]:


fitnessFunctionCircle_EnergyMulti = singleAsMultiValue(fitnessFunctionCircle_Energy)
fitnessFunctionCircle_DistanceMulti = singleAsMultiValue(fitnessFunctionCircle_Distance)
fitnessFunctionCircle_FullResultsMulti = singleAsMultiValue(fitnessFunctionCircle_FullResults)


# #### Demonstration

# In[ ]:


def localDemo():
    results = fitnessFunctionCircle_FullResults([4])
    #print(results)
    plt.plot(results['x'], results['y'])
    
if _runDemos:
    localDemo()


# In[ ]:


def localDemo():
    results = fitnessFunctionCircle_FullResultsMulti([[2], [16]])
    plt.plot(results[0]['x'], results[0]['y'])
    plt.plot(results[1]['x'], results[1]['y'])
    
if _runDemos:
    localDemo()


# ### Robins

# #### Singlevalue Functions

# In[ ]:


fitnessFunctionRobins_Energy = createFitnessFunction(simulationDescription, fromChromozomeToDescriptionCircle, createCircleControllerWithGain, energySelector)
fitnessFunctionRobins_Distance = createFitnessFunction(simulationDescription, fromChromozomeToDescriptionCircle, createCircleControllerWithGain, distanceSelector)
fitnessFunctionRobins_FullResults = createFitnessFunction(simulationDescription, fromChromozomeToDescriptionCircle, createCircleControllerWithGain)


# #### Multivalue Functions

# In[ ]:


fitnessFunctionRobins_EnergyMulti = singleAsMultiValue(fitnessFunctionRobins_Energy)
fitnessFunctionRobins_DistanceMulti = singleAsMultiValue(fitnessFunctionRobins_Distance)
fitnessFunctionRobins_FullResultsMulti = singleAsMultiValue(fitnessFunctionRobins_FullResults)


# ## Computer Cluster Help

# Multivalue fitness function is function which takes an array of chromozomes and returns another array containing the fitness values for all given chromozomes. Body of a such function could be implemented as a parallel process which decrease time needed for its evaluation.
# 
# The parallel process might be implemented in different ways. One of them, and probably the best one, is usage of distributed evaluation with help of computer cluster. Computer cluster creation is well-documented process with standard steps, thus, a scientist who want to use this technique must just define single environment, environment for evaluation of single fitness function. 
# 
# As an envelope around this environment the web service might be used. However, this leads to asynchronous execution. As the optimization libraries for Python are synchronous, the connection between asynchronous and synchronous parts must be created.

# ### Main Function for Server

# In[ ]:


def evaluateSingleFLCSimulation(description):
    result = runSimulation(description, createFuzzyController, selectors)
    return result

