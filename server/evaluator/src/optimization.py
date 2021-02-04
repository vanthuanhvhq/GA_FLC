#!/usr/bin/env python
# coding: utf-8

import numpy as np
import math
import random
from math import *
import scipy.integrate as integrate
import matplotlib.pyplot as plt
import pandas as pd
#get_ipython().run_line_magic('matplotlib', 'inline')
import skfuzzy as fuzz


# ## Robot parameters

# In[31]:


#r = 0.0925
#b = 0.37
#m = 9  
#I = 0.16245
#start_x = 0
#start_y = 0
#start_theta = -45
#start_theta = (start_theta * math.pi) / 180

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


# ## Kinematic model

# In[32]:


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
        #E = velocity * omega * sqrt(m*I)
        return [x_dot, y_dot, theta_dot, velocity, omega, E] #velocity, omega, E are returned for easy evaluation they are not needed for computation
    return robot

robot = createRobot(robotParams)


# ## Dynamic Model

# In[55]:


#//////////////////////////////////////////////////////////////////////////////
# Calculating the coefficients for DC motor
#J = 0.01
#B = 0.1
#Kt = Ke = K = 0.01
#Ra = 0.1
#La = 0.01
#b1 = K/(La*J)
#b0 = 0
#a1 = -(Ra*B + K*K)/(La*J)
#a0 = -(Ra*J + La*B)/(La*J)

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
    #motorAsFilter = createFilter2ndOrder(b1, b0, a1, a0)
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
        #E = velocity * omega * sqrt(m*I)
        return [x_dot, y_dot, theta_dot, velocity, omega, E, *vLStateD, *vRStateD] #velocity, omega, E are returned for easy evaluation they are not needed for computation

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
        #E = velocity * omega * sqrt(m*I)
        return [x_dot, y_dot, theta_dot, velocity, omega, E] #velocity, omega, E are returned for easy evaluation they are not needed for computation

    if motorModel is None:
        return robot
    else:
        return robotWithDynamic
    pass
  
motorModel = createMotorModel(motorParams)    
robotWithDynamic = createRobotModelWithDynamic(robotParams, motorModel)
robot = robotWithDynamic # If delete this line, only the Kinematic model is taken into account.


# ## Solver

# In[73]:


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

# In[57]:


def controllerForPath(controller, path):
    destinationX, destinationY, destinationOrietation = next(path)
    destinationState = [destinationX, destinationY, destinationOrietation]
    lastReached = False
  #print('Destination set to', destinationState)
    def result(t, currentState):
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
          if (deltaX * deltaX + deltaY * deltaY < 1e-2):
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

# In[58]:


def robotModelCreator(robotModel, controllerCreator, path, **kwargs):
    controller_ = controllerCreator(**kwargs)
    savedController = controllerForPath(controller_, path)
    def resultRMC(t, currentState):
        return robotModel(t, currentState, savedController)
    return resultRMC


# In[59]:


def simpleCompute(computefunc, state0, t0 = 0, t_bound = 200, max_step = 0.05):
    def resultSC(model):
        return computefunc(
          model, state0 = state0, t0 = t0, t_bound = t_bound, max_step = max_step)
    return resultSC


# ## Fully parametrized fitness function

# In[74]:


fuzzyDescription = {
    'inputs': {
        'distance' : {
            'S': np.arange(-3.14, 3.14, 0.0628),
            'M': {'VC': [0, 0, 1], 'C': [0, 1, 2], 'M': [1, 2, 3], 'F': [2, 3, 4], 'VF': [3, 4, 4]}
        },
        'angle' : {
            'S': np.arange(-3.14, 3.14, 0.0628),
            'M': {'BN': [-3.14, -3.14, -1.57], 'N': [-3.14, -1.57, 0], 'Z': [-1.57, 0, 1.57], 'P': [0, 1.57, 3.14], 'BP': [1.57, 3.14, 3.14]}
        }
    },
    'outputs': {
        'omegaR': {
            'S': np.arange(0, 30, 0.3),
            'rules': {
                'VC': {'BN': 'VSR', 'N': 'SR', 'Z': 'VSR', 'P': 'BR', 'BP': 'VBR'},
                'C': {'BN': 'VSR', 'N': 'SR', 'Z': 'SR', 'P': 'BR', 'BP': 'VBR'},
                'M': {'BN': 'VSR', 'N': 'SR', 'Z': 'MBR', 'P': 'BR', 'BP': 'VBR'},
                'F': {'BN': 'VSR', 'N': 'SR', 'Z': 'BR', 'P': 'BR', 'BP': 'VBR'},
                'VF': {'BN': 'VSR', 'N': 'SR', 'Z': 'VBR', 'P': 'BR', 'BP': 'VBR'}
            },
            'mode': 'centroid',
            'M': {'VSR': [0, 0, 12], 'SR': [0, 6, 12], 'MBR': [6, 12, 18], 'BR': [12, 18, 24], 'VBR': [18, 24, 30]}
        },
        'omegaL': {
            'S': np.arange(0, 30, 0.3),
            'rules': {
                'VC': {'BN': 'VBL', 'N': 'BL', 'Z': 'VSL', 'P': 'SL', 'BP': 'VSL'},
                'C': {'BN': 'VBL', 'N': 'BL', 'Z': 'SL', 'P': 'SL', 'BP': 'VSL'},
                'M': {'BN': 'VBL', 'N': 'BL', 'Z': 'MBL', 'P': 'SL', 'BP': 'VSL'},
                'F': {'BN': 'VBL', 'N': 'BL', 'Z': 'BL', 'P': 'SL', 'BP': 'VSL'},
                'VF': {'BN': 'VBL', 'N': 'BL', 'Z': 'VBL', 'P': 'SL', 'BP': 'VSL'} 
            },
            'mode': 'centroid',
            'M': {'VSL': [0, 0, 12], 'SL': [0, 6, 12], 'MBL': [6, 12, 18], 'BL': [12, 18, 24], 'VBL': [18, 24, 30]}
        }
    }
}



def FuzzyLogic_Using_For_GA(fuzzyDescription, r, b, omega_ri, vri,lowVelocityLimit, highVelocityLimit, lowOmegaLimit, highOmegaLimit):
    inputsDistance = fuzzyDescription['inputs']['distance']['M']
    inputsSpaceDistance = fuzzyDescription['inputs']['distance']['S']
    inputsAngle = fuzzyDescription['inputs']['angle']['M']
    inputsSpaceAngle = fuzzyDescription['inputs']['angle']['S']
    outputsOmegaR = fuzzyDescription['outputs']['omegaR']['M']
    outputSpaceOmegaR = fuzzyDescription['outputs']['omegaR']['S']
    outputsOmegaL = fuzzyDescription['outputs']['omegaL']['M']
    outputSpaceOmegaL = fuzzyDescription['outputs']['omegaL']['S']
    outputRulesOmegaR = fuzzyDescription['outputs']['omegaR']['rules']
    outputRulesOmegaL = fuzzyDescription['outputs']['omegaL']['rules']
    #outputsOmegaR
    def createFuzzyfier(space, categories, trimf = fuzz.trimf, membership = fuzz.interp_membership):
        space = np.array(space)
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
        outputSpace = np.array(outputSpace)
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
        outputSpace = np.array(outputSpace)
        def result(value):
            return defuzz(outputSpace, value, *defuzzArgs, **defuzzKwargs)
        return result
    def createFullFuzzySystem(inferenceSystem, defuzzyfier):
        def system(inputA, inputB):
            return defuzzyfier(inferenceSystem(inputA, inputB))
        return system

    inputsDistanceFuzzyfier = createFuzzyfier(inputsSpaceDistance, inputsDistance)
    inputsAngleFuzzyfier = createFuzzyfier(inputsSpaceAngle, inputsAngle)

    inferenceSystem_R = createInferenceSystem(inputsDistanceFuzzyfier, inputsAngleFuzzyfier, outputSpaceOmegaR, outputsOmegaR, outputRulesOmegaR)
    outputDefuzzyfier_R = createDefuzzyfier(outputSpaceOmegaL, mode=fuzzyDescription['outputs']['omegaR']['mode'])

    inferenceSystem_L = createInferenceSystem(inputsDistanceFuzzyfier, inputsAngleFuzzyfier, outputSpaceOmegaL, outputsOmegaL, outputRulesOmegaL)
    outputDefuzzyfier_L = createDefuzzyfier(outputSpaceOmegaL, mode=fuzzyDescription['outputs']['omegaL']['mode'])

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

FuzzyLogic_Using_For_GA(fuzzyDescription, r=1, b=1, **{
        'omega_ri': 0, 'vri': 1.0,'lowVelocityLimit': 0.2, 
        'highVelocityLimit': 2.0, 'lowOmegaLimit': -0.75, 'highOmegaLimit': 0.75
    })    


# In[75]:


fitnessFunctionDescription = {
    'fuzzyDescription': fuzzyDescription,
    'robotState0': {
        'x': 0,
        'y': 0,
        'theta': -3.14 / 4
    },

    'path': [
        [0, 0, 0],  #X, Y, orientation
        [10, 0, 0], #X, Y, orientation
        [10, 10, 0], #X, Y, orientation
        [0, 10, 0], #X, Y, orientation
        [0, 0, 0]
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
        'omega_ri': 0, 'vri': 1.0,'lowVelocityLimit': 0.2, 
        'highVelocityLimit': 2.0, 'lowOmegaLimit': -0.75, 'highOmegaLimit': 0.75
    },
}


# In[76]:


fitnessFunctionDescription = {
    'fuzzyDescription': {
        'inputs': {
            'distance' : {
                'S': np.arange(-3.14, 3.14, 0.0628),
                'M': {'VC': [0, 0, 1], 'C': [0, 1, 2], 'M': [1, 2, 3], 'F': [2, 3, 4], 'VF': [3, 4, 4]}
            },
            'angle' : {
                'S': np.arange(-3.14, 3.14, 0.0628),
                'M': {'BN': [-3.14, -3.14, -1.57], 'N': [-3.14, -1.57, 0], 'Z': [-1.57, 0, 1.57], 'P': [0, 1.57, 3.14], 'BP': [1.57, 3.14, 3.14]}
            }
        },
        'outputs': {
            'omegaR': {
                'S': np.arange(0, 30, 0.3),
                'rules': {
                    'VC': {'BN': 'VSR', 'N': 'SR', 'Z': 'VSR', 'P': 'BR', 'BP': 'VBR'},
                    'C': {'BN': 'VSR', 'N': 'SR', 'Z': 'SR', 'P': 'BR', 'BP': 'VBR'},
                    'M': {'BN': 'VSR', 'N': 'SR', 'Z': 'MBR', 'P': 'BR', 'BP': 'VBR'},
                    'F': {'BN': 'VSR', 'N': 'SR', 'Z': 'BR', 'P': 'BR', 'BP': 'VBR'},
                    'VF': {'BN': 'VSR', 'N': 'SR', 'Z': 'VBR', 'P': 'BR', 'BP': 'VBR'}
                },
                'mode': 'centroid',
                'M': {'VSR': [0, 0, 12], 'SR': [0, 6, 12], 'MBR': [6, 12, 18], 'BR': [12, 18, 24], 'VBR': [18, 24, 30]}
            },
            'omegaL': {
                'S': np.arange(0, 30, 0.3),
                'rules': {
                    'VC': {'BN': 'VBL', 'N': 'BL', 'Z': 'VSL', 'P': 'SL', 'BP': 'VSL'},
                    'C': {'BN': 'VBL', 'N': 'BL', 'Z': 'SL', 'P': 'SL', 'BP': 'VSL'},
                    'M': {'BN': 'VBL', 'N': 'BL', 'Z': 'MBL', 'P': 'SL', 'BP': 'VSL'},
                    'F': {'BN': 'VBL', 'N': 'BL', 'Z': 'BL', 'P': 'SL', 'BP': 'VSL'},
                    'VF': {'BN': 'VBL', 'N': 'BL', 'Z': 'VBL', 'P': 'SL', 'BP': 'VSL'} 
                },
                'mode': 'centroid',
                'M': {'VSL': [0, 0, 12], 'SL': [0, 6, 12], 'MBL': [6, 12, 18], 'BL': [12, 18, 24], 'VBL': [18, 24, 30]}
            }
        }
    },
    
    'robotState0': {
        'x': 0,
        'y': 0,
        'theta': -3.14 / 4
    },

    'path': [
        [0, 0, 0],  #X, Y, orientation
        [10, 0, 0], #X, Y, orientation
        [10, 10, 0], #X, Y, orientation
        [0, 10, 0], #X, Y, orientation
        [0, 0, 0]
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
        'omega_ri': 0, 'vri': 1.0,'lowVelocityLimit': 0.2, 
        'highVelocityLimit': 2.2, 'lowOmegaLimit': -0.75, 'highOmegaLimit': 0.75
    },
}


# In[81]:


def fullParametrizedFitnessFunction(fitnessFunctionDescription):
    pathForSimulationAsArray = fitnessFunctionDescription['path']
    robotParams = fitnessFunctionDescription['robotParams']
    t0 = 0
    t_bound = 60
    max_step = 0.05
    state0 = None
    
    robotState0 = fitnessFunctionDescription['robotState0']
    if robotParams['motorParams'] is None:
        state0 = np.array([robotState0['x'], robotState0['y'], robotState0['theta'], 0, 0, 0]) # x0=0, y0=0,theta
    else:
        state0 = np.array([robotState0['x'], robotState0['y'], robotState0['theta'], 0, 0, 0, 0, 0, 0, 0]) # x0=0, y0=0, theta

    solverfunc = simpleCompute(
      compute, state0 = state0, 
      t0 = t0, t_bound = t_bound, max_step = max_step)

    def functionForEvaluation(results): # to optimize energy
        lastResult = results[-1]
        #return lastResult[-1][-1]
        return lastResult

    def createRobot(robotParams):
        motorModel = createMotorModel(robotParams['motorParams'])
        robotModel = createRobotModelWithDynamic(robotParams, motorModel)
        r = robotParams['r']
        b = robotParams['b']
        
        pathForSimulation = iter(pathForSimulationAsArray)
        controllerParams_Fuzzy_GA = {'fuzzyDescription': fitnessFunctionDescription['fuzzyDescription'], 'r': r, 'b': b, **fitnessFunctionDescription['controllerParams']}
        robot = robotModelCreator(robotModel, FuzzyLogic_Using_For_GA, pathForSimulation, **controllerParams_Fuzzy_GA)
        return robot

    def runSimulation(robot):
        results = []
        robotStates = solverfunc(robot)      
        for currentState in robotStates: # readout all states from current moving robot
            pass
        result = {}
        for key, func in selectors.items():
            result[key] = func(currentState)
        results.append(result)
        return results

    fullRobotModel = createRobot(robotParams)

    results = runSimulation(fullRobotModel)
    value = functionForEvaluation(results)
    return value


# In[82]:


#fitnessFunctionValue = fullParametrizedFitnessFunction(fitnessFunctionDescription)
#print(fitnessFunctionValue)


# In[ ]:




