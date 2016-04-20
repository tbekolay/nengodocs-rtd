
# coding: utf-8

# # The `config` system
# 
# Nengo objects have many parameters
# that can be modified.
# Some of these parameters
# are critical characteristics of that object,
# and others are hints or suggestions
# that a backend can use or ignore.
# 
# Nengo's `config` system is designed
# to make setting large numbers of parameters easy,
# and to allow backends
# to introduce additional parameters
# without changing core Nengo objects.

# In[ ]:

from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt

import nengo
from nengo.utils.functions import piecewise
from nengo.utils.ipython import hide_input
hide_input()


# ## Setting default parameters
# 
# The `config` system aids in
# setting many parameters
# with a **hierarchy of defaults**.
# When you create a Nengo object,
# any parameters not specified
# will be given the value `nengo.Default`.
# This value tells Nengo
# to use the default value
# with the highest precendence
# in the hierarchy.
# Every `Network` has an associated
# `config` object,
# on which defaults can be set.
# The network hierarchy is traversed
# from most to least specific
# and the first network with a default
# set for that particular parameter
# is used. For example:
# 
#     with nengo.Network() as net:
#         with nengo.Network() as subnet:
#             with nengo.Network() as subsubnet:
#                 ens = nengo.Ensemble(10, 1)
# 
# When filling in defaults for `ens`,
# the hierarchy looks like
# 
#     └── net                <- least specific
#         └── subnet
#             └── subsubnet  <- most specific
# 
# so defaults set in `subsubnet`
# will take precedence over those in `subnet`,
# which take precedence over those in `net`.
# 
# If no default has been set in the
# network hierarchy,
# then the parameter default
# is used.
# These defaults are specified
# when the Nengo objects are created.
# We can investigate these defaults
# by printing the class attributes
# associated with them.

# In[ ]:

# Get all info about the radius
print(nengo.Ensemble.radius)
# Just get the default
print(nengo.Ensemble.radius.default)


# We can inspect which defaults
# have been overridden in a
# particular `config` object
# by printing it.
# 

# In[ ]:

model = nengo.Network()
print(model.config)


# To configure a parameter
# (i.e., change its network-local default),
# set it as shown below.

# In[ ]:

model.config[nengo.Ensemble].radius = 1.5
print(model.config[nengo.Ensemble])


# Within this network, the default radius will be 1.5.

# In[ ]:

with nengo.Network():
    ens = nengo.Ensemble(10, 1)
print("Normal network: ens.radius = %s" % ens.radius)

with model:
    ens = nengo.Ensemble(10, 1)
print("Configured network: ens.radius = %s" % ens.radius)


# Note that if a radius is explicitly passed in,
# it will always be used.

# In[ ]:

with nengo.Network():
    ens = nengo.Ensemble(10, 1, radius=2.0)
print("Normal network: ens.radius = %s" % ens.radius)

with model:
    ens = nengo.Ensemble(10, 1, radius=2.0)
print("Configured network: ens.radius = %s" % ens.radius)


# When networks are nested within one another,
# the most specific network configuration is used.
# For example, if you create an Ensemble
# without specifying a radius,
# it will first check the network
# that the Ensemble is a part of;
# if that network has not configured a default,
# then it will check the network
# that that network is part of,
# and so on.

# In[ ]:

with model:
    
    with nengo.Network() as subnet:
        subnet.config[nengo.Ensemble].neuron_type = nengo.LIFRate()
        
        with nengo.Network() as subsubnet:
            subsubnet.config[nengo.Ensemble].radius = 2.0
            print("Creating e1 in subsubnet")
            e1 = nengo.Ensemble(10, 1)
            # Uses subsubnet.config value for radius
            print("  radius =", e1.radius)
            # Uses subnet.config value for neuron_type
            print("  neuron_type =", e1.neuron_type)

        print("Creating e2 in subnet")
        e2 = nengo.Ensemble(10, 1)
        # Uses model.config value for radius
        print("  radius =", e2.radius)
        # Uses subnet.config value for neuron_type
        print("  neuron_type =", e2.neuron_type)
        
    print("Creating e3 in model")
    e3 = nengo.Ensemble(10, 1)
    # Uses model.config value for radius
    print("  radius =", e3.radius)
    # Uses nengo.Ensemble default for neuron_type
    print("  neuron_type =", e3.neuron_type)


# Note that each `config` object
# only knows about the defaults set on itself.

# In[ ]:

with model:
    with nengo.Network() as subnet:
        subnet.config[nengo.Ensemble].neuron_type = nengo.LIFRate()
        with nengo.Network() as subsubnet:
            subsubnet.config[nengo.Ensemble].radius = 2.0
print("subsubnet:")
print(subsubnet.config[nengo.Ensemble])
print("\nsubnet:")
print(subnet.config[nengo.Ensemble])
print("\nmodel:")
print(model.config[nengo.Ensemble])


# If you want a more global picture of the defaults
# in the *current context*, you can query the `Config`
# class itself (all `config` objects are instances of `Config`).
# 
# To query all parameters, print `Config.all_defaults()`.
# You may pass a Nengo object class to this function
# to filter the results.
# For example, to get all defaults set for `Ensemble`,
# use `Config.all_defaults(nengo.Ensemble)`.

# In[ ]:

with model:
    print("In 'model' context:")
    print(nengo.Config.all_defaults())

    with nengo.Network() as subnet:
        subnet.config[nengo.Ensemble].neuron_type = nengo.LIFRate()
        subnet.config[nengo.Ensemble].radius = 3.0
        print("\nIn 'subnet' context:")
        print(nengo.Config.all_defaults(nengo.Ensemble))

        with nengo.Network() as subsubnet:
            subsubnet.config[nengo.Ensemble].neuron_type = nengo.Direct()
            subsubnet.config[nengo.Ensemble].radius = 2.0
            print("\nIn 'subsubnet' context:")
            print(nengo.Config.all_defaults(nengo.Ensemble))


# The default value for a particular parameter
# can be queried from the global context
# with the `nengo.Config.default` function.

# In[ ]:

help(nengo.Config.default)


# In[ ]:

def print_defaults():
    def_radius = nengo.Config.default(nengo.Ensemble, 'radius')
    def_type = nengo.Config.default(nengo.Ensemble, 'neuron_type')
    print("  default radius: %s" % def_radius)
    print("  default neuron_type: %s" % def_type)
 
with model:
    with nengo.Network() as subnet:
        subnet.config[nengo.Ensemble].neuron_type = nengo.LIFRate()
        with nengo.Network() as subsubnet:
            subsubnet.config[nengo.Ensemble].radius = 2.0
            print("subsubnet:")
            print_defaults()
        print("\nsubnet:")
        print_defaults()
    print("\nmodel:")
    print_defaults()


# #### Defaults are filled in immediately
# 
# One important feature about the defaults hierarchy
# is that defaults are filled in **immediately**.
# When you create a Nengo object,
# the attributes are filled in with the **current**
# defaults that are set.
# Changing the defaults after object creation
# will not update objects already created.

# In[ ]:

with model:
    e1 = nengo.Ensemble(10, 1)
    print("e1.radius =", e1.radius)
    print("Changing default radius to 2.0")
    model.config[nengo.Ensemble].radius = 2.0
    e2 = nengo.Ensemble(10, 1)
    print("e1.radius =", e1.radius)
    print("e2.radius =", e2.radius)


# #### Resetting to default
# 
# If you ever wish to reset a value
# back to the default,
# you can remove it from the `config` object you modified.

# In[ ]:

with model:
    e1 = nengo.Ensemble(10, 1)
    print("e1.radius =", e1.radius)
    print("Resetting radius back to default")
    del model.config[nengo.Ensemble].radius
    print("\n" + str(model.config[nengo.Ensemble]) + "\n")
    e2 = nengo.Ensemble(10, 1)
    print("e2.radius =", e2.radius)


# #### Making new `config`s
# 
# Typically, several Nengo objects
# will share a set of parameters,
# but won't make sense to encapsulate in a network.
# One method of having those objects share parameters
# is to use dictionary unpacking.

# In[ ]:

with nengo.Network():
    hippocampus_args = {'radius': 1.5, 'neuron_type': nengo.LIFRate()}
    e1 = nengo.Ensemble(100, 2, **hippocampus_args)
    e2 = nengo.Ensemble(150, 3, **hippocampus_args)
    e3 = nengo.Ensemble(200, 4, **hippocampus_args)
print(e1.radius, e2.radius, e3.radius)


# An alternative method that can be very useful
# for large networks and for more readable models
# is to create a new `config` object
# to encapsulate those parameter settings.

# In[ ]:

in_hippocampus = nengo.Config(nengo.Ensemble)
in_hippocampus[nengo.Ensemble].radius = 1.5
in_hippocampus[nengo.Ensemble].neuron_type = nengo.LIFRate()

with nengo.Network():
    with in_hippocampus:
        e1 = nengo.Ensemble(100, 2)
        e2 = nengo.Ensemble(150, 3)
        e3 = nengo.Ensemble(200, 4)
print(e1.radius, e2.radius, e3.radius)


# ## Advanced: adding new parameters
# 
# This section is targetted to those
# implementing new backends
# or large libraries of networks
# (like, for example, `nengo.SPA`).
# 
# Often, you want to associate some kind of
# metadata with a Nengo object,
# or a type of Nengo objects.
# For example, in backends
# that communicate with specific hardware,
# it can be helpful to mark certain nodes
# as being time-dependent,
# or to assign certain ensembles
# to a particular portion of the hardware memory.
# 
# Python allows us to make new attributes
# on Nengo objects.
# However, we highly discourage this activity,
# because a Nengo object should be
# a backend-agnostic part of a model.
# The parameters pre-defined on Nengo objects
# make up the parameters that all backends
# should deal with in some way.
# 
# For this reason, we raise a warning
# when creating a new attribute.

# In[ ]:

with nengo.Network():
    ens = nengo.Ensemble(10, 1)
    ens.memory_location = 0x1000


# So how should backends associate arbitrary information
# with Nengo objects?
# The `config` system!
# 
# We saw above that we can create new `config` objects
# by specifying which Nengo objects they can configure.
# We can also create new parameters
# on those `config` objects.

# In[ ]:

import nengo.params
my_config = nengo.Config(nengo.Ensemble)
# memory_location must be a positive integer
my_config[nengo.Ensemble].set_param(
    'memory_location',
    nengo.params.IntParam('memory_location', default=None, optional=True, low=0))


# Now, we can set that parameter
# for the `nengo.Ensemble` class as a whole,
# or with individual instances.

# In[ ]:

# Make the network (this code is backend-agnostic)
with nengo.Network():
    e1 = nengo.Ensemble(10, 1)
    e2 = nengo.Ensemble(10, 1)

# Set backend-specific parameters
my_config[nengo.Ensemble].memory_location = 0x1000  # Set Ensemble default
my_config[e2].memory_location = 0x2000  # Set value for e2

print('e1 will be stored at 0x%x' % my_config[e1].memory_location)
print('e2 will be stored at 0x%x' % my_config[e2].memory_location)


# `Parameter` types for the most common Python objects
# are available in `nengo.params`,
# as well as other types that Nengo uses frequently,
# but it is possible to implement your own
# in order to do additional processing
# like validation.
# See the `nengo.params` source
# for examples.

# In[ ]:

[cls for cls in dir(nengo.params) if cls.endswith('Param')]

