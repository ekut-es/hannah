import torch
try:
    import Levenshtein as Lev
except:
    print("Could not import Levenshtein distance")
    pass
    
class Decoder(object):
    def __init__(self, labels, empty=0):
        """ Constructor of decoder 

        Arguments
        ---------
        labels : dict
          Dict from class label to class name
        empty : int
          Label of the empty class
        """

        self.labels = labels
        self.empty = empty

    def _decode_one(self, scores):
        new_scores = []
        last = None
        for item in scores:
            if item == self.empty:
                continue
            if item == last:
                continue

            last = item
            new_scores.append(item)
            
        return new_scores
        

    def decode(self, scores, lengths):
        """ 
        Decodes the given Tensor of predictions, for now uses greedy decoding.
      
        Predictions are shortend to lengths, empty label is removed 
        and repeted predictions of the same class are removed.

        Arguments
        ---------

        scores : torch.Tensor
           Tensor of format (max_length x batch_size x nr_classes) 
           with softmax values for each class 
     
        in_lengths : torch.Tensor(int)
           Tensor of ints (<= max_length) with the actual lengths of the predictions
        """

        max_scores = torch.max(scores.permute(1,0,2), 2)[1]

        
    def calculate_error(self, scores, score_lengths, labels, label_lengths):
        """ Calculate error"""

        max_scores = torch.max(scores.permute(1,0,2), 2)[1]

        sum_error = 0.0
        sum_cer = 0.0
        num_predictions = max_scores.size(0) 

        print("Error calculation")
        for i in range(num_predictions):
            current_scores = max_scores[i].tolist()
            print(current_scores)
            current_scores = current_scores[:int(score_lengths[i].item())]
            current_labels = labels[i].tolist()
            current_labels = current_labels[:int(label_lengths[i].item())]

            current_scores = self._decode_one(current_scores)
            current_labels = self._decode_one(current_labels)

            error = Lev.distance("".join(chr(e) for e in  current_scores),
                                 "".join(chr(e) for e in  current_labels))

            sum_cer += error
            
            error /= max(len(current_scores), len(current_labels), 1.0)

            print(current_scores, current_labels)
            print("Errors: ", error)
            
            sum_error += error
            
        avg_error = sum_error / num_predictions
        avg_cer = sum_cer / num_predictions
        return avg_error, sum_cer
