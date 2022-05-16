import xenodiffusionscope

#This file needs to be properly expanded. 
#Curretnly only tries to make objects to test the classes.

def test_TPC():
    radius, length, liquid_gap, gas_gap, drift_field = 15,2600,5,5,100
    test_TPC = xenodiffusionscope.TPC(radius, length, liquid_gap, gas_gap, drift_field)
    return test_TPC

def test_MeshGrid():
    r_max, hex_side = 15,1.5
    test_MeshGrid = xenodiffusionscope.MeshGrid(r_max, hex_side)
    return test_MeshGrid

def test_XeLamp():
    delta_t_lamp = 0.1
    test_XeLamp = xenodiffusionscope.XeLamp(delta_t_lamp)
    return test_XeLamp

def test_ElectronDrift():
    tpc, xelamp,drift_delta_t = test_TPC(), test_XeLamp, 1
    test_ElectronDrift = xenodiffusionscope.ElectronDrift(tpc, xelamp,drift_delta_t)
    return test_ElectronDrift

def test_LCEPattern():
    tpc = test_TPC()
    test_LCEPattern = xenodiffusionscope.LCEPattern(tpc)
    return test_LCEPattern

def get_toparray_models():
    '''
    Get the model names from directory "TopArrayModel"
    '''
    pass

def test_TopArray():
    tpc, mesh,model,path_to_patterns = test_TPC(), test_MeshGrid(), '/some/path/'
    test_TopArray(tpc, mesh,model,path_to_patterns) 
    #this doesn't actually test anything useful... for now?

    return test_TopArray