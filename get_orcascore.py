import synapseclient
import synapseutils

syn = synapseclient.Synapse()
syn.login(authToken="")
files = synapseutils.syncFromSynapse(syn, 'syn18824258',path='/Users/mcgoug01/Downloads/orcascore/')