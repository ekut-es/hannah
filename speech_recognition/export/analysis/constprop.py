import numpy as np
from collections import defaultdict

class ConstPropState(object):
    def __init__ (self, value = None, shape = None):
        self.value = value
        self.shape = shape
        
    def __str__(self):
        return "shape: " + str(self.shape) + " " + "value: " + str(self.value)

    def __repr__(self):
        return "shape: " + str(self.shape) + " " + "value: " + repr(self.value)

        
    def __iter__(self):
        return iter((self.value, self.shape))
            
    def __eq__(self, other):
        if type(self) != type(other):
            return False

        values = False
        if self.value is None and other.value is None:
            values = True
        elif type(self.value) == np.ndarray and type(other.value) == np.ndarray:
            values = (self.value == other.value).all()
            
        return  self.shape == other.shape and values



def constant_propagation(graph):
    """Identifie constant values and shapes using constant propagation"""
    state_dict = defaultdict(OutputState)
    worklist = []

    for node in graph.nodes:
        worklist.append(node)
    
    while worklist:
        node = worklist[0]
        worklist = worklist[1:]
        attrs = node.attrs
        changed = False

        if node.op_type == 'Shape':
            input = node.inputs[0]
            input_value, input_shape = state_dict[input]
            out = OutputState(None, None)
            if input_shape is not None:
                out = OutputState(np.array(input_shape), np.array(input_shape).shape)
            output = node.outputs[0]
            if state_dict[output] != out:
                changed=True
                state_dict[output] = out
                
        elif node.op_type == 'Constant':
            out = OutputState(attrs['value'], attrs['value'].shape)
            output = node.outputs[0]
            
            if out != state_dict[output]:
                state_dict[output] = out
                changed=True
                
        elif node.op_type == 'Gather':
            data = node.inputs[0]
            indices = node.inputs[1]
            axis = attrs['axis']
            
            data_state, data_shape = state_dict[data]
            index_state, index_shape = state_dict[indices]


            def calc(data, index):
                out = []
                y = np.take(data, index, axis=axis)
                return y

            out = OutputState(None, None)
            if data_state is not None and index_state is not None:
                res = calc(data_state, index_state)
                out = OutputState(res, res.shape)
            elif index_state is not None and data_shape is not None:
                dummy_data = np.zeros(data_shape)
                res = calc(dummy_data, index_state)
                out = OutputState(None, res.shape)

            o = node.outputs[0]
            if out != state_dict[o]:
                state_dict[o] = out
                changed = True
                
        elif node.op_type == 'Unsqueeze':
            axes = attrs['axes']
 
            out = OutputState(None, None)
            data_state, data_shape = state_dict[node.inputs[0]]

            
            if data_state is None and data_shape is not None:
                data = np.zeros(data_shape)
            else:
                data = data_state

            if len(axes) == 1:
                axes = axes[0]
                
            if data is not None:               
                res = np.expand_dims(data, axis=axes)

                if data_state is not None:
                    out = OutputState(res, res.shape)
                elif data_shape is not None:
                    out = OutputState(None, res.shape)
                    
         
            if out != state_dict[node.outputs[0]]:
                state_dict[node.outputs[0]] = out
                changed = True
            
        elif node.op_type == 'Concat':
            axis = attrs['axis']

            input_states = []
            input_shapes = []
            for i in node.inputs:
                input_states.append(state_dict[i].value)
                input_shapes.append(state_dict[i].shape)
            
 
        
            out = OutputState(None, input_shapes[0] if None not in input_shapes else None)
            if None not in input_states:
                res = np.concatenate(input_states, axis=axis)
                out = OutputState(res, res.shape)
                
            if state_dict[node.outputs[0]] != out:
                state_dict[node.outputs[0]] = out
                changed = True
            
        elif node.op_type == 'Reshape':
            input_shape = state_dict[node.inputs[0]].shape
            input_data = state_dict[node.inputs[1]].value

            out = OutputState(None, None)

            if input_shape is not None and input_data is not None:
                data = np.zeros(input_shape)
                res = np.reshape(data, input_data)
                out = OutputState(None, res.shape)
            
            if state_dict[node.outputs[0]] != out:
                state_dict[node.outputs[0]] = out
                changed = True

        elif node.op_type == 'Gemm':
            input_shape = state_dict[node.inputs[0]].shape
            out = OutputState(None, None)

            matrix_shape = node.input_tensors[node.inputs[1]].shape
            
            if input_shape is not None:
                out = OutputState(None, (input_shape[0], matrix_shape[0]))

            if state_dict[node.outputs[0]] != out:
                state_dict[node.outputs[0]] = out
                changed = True
                
        elif node.op_type == "Relu":
            input_shape = state_dict[node.inputs[0]].shape
            out = OutputState(None, input_shape)

            if state_dict[node.outputs[0]] != out:
                state_dict[node.outputs[0]] = out
                changed = True
        else:
            raise Exception("ConstProp: Unhandled op {} using shape_dict"-format(node.op_type))
            changed = False
            for o in node.outputs:
                out = OutputState(None, None)
            
                if o in graph.shape_dict:
                    out = OutputState(None, graph.shape_dict[o])

                
                    
                if out != state_dict[o]:
                    changed = True
                    state_dict[o] = out

        if changed:
            for child in node.children:
                if not child in worklist:
                    worklist.append(child)
        
    return state_dict
