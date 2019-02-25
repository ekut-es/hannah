import distiller
import apputils


def draw_classifier_to_file(model, png_fname, input, display_param_nodes=False, rankdir='TB', styles=None):
    """Draw a PyTorch classifier to a PNG file.  This a helper function that
    simplifies the interface of draw_model_to_file().

    Args:
        model: PyTorch model instance
        png_fname (string): PNG file name
        input (tensor): one batch of input_data
        display_param_nodes (boolean): if True, draw the parameter nodes
        rankdir: diagram direction.  'TB'/'BT' is Top-to-Bottom/Bottom-to-Top
                 'LR'/'R/L' is Left-to-Rt/Rt-to-Left
        styles: a dictionary of styles.  Key is module name.  Value is
                a legal pydot style dictionary.  For example:
                styles['conv1'] = {'shape': 'oval',
                                   'fillcolor': 'gray',
                                   'style': 'rounded, filled'}
    """

    try:
        model = distiller.make_non_parallel_copy(model)
        g = apputils.SummaryGraph(model, input)
        apputils.draw_model_to_file(g, png_fname, display_param_nodes, rankdir, styles)
        print("Network PNG image generation completed")
    except FileNotFoundError:
        print("An error has occured while generating the network PNG image.")
        print("Please check that you have graphviz installed.")
        print("\t$ sudo apt-get install graphviz or")
        print("\t$ sudo yum install graphviz")

