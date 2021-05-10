# Profiling Tools


To debug training inefficiencies we support the following profiling tools:

## Simple Profiler:

     hannah-train profiler=simple trainer.max_epochs=1 profiler=simple

Generates a summary of the durations of the individual training steps:


     Action                             	|  Mean duration (s)	|Num calls      	|  Total time (s) 	|  Percentage %   	|
     ------------------------------------------------------------------------------------------------------------------------------------
     Total                              	|  -              	|_              	|  29.42          	|  100 %          	|
     ------------------------------------------------------------------------------------------------------------------------------------
     run_training_epoch                 	|  21.692         	|1              	|  21.692         	|  73.733         	|
     run_training_batch                 	|  0.030893       	|288            	|  8.8971         	|  30.241         	|
     optimizer_step_and_closure_0       	|  0.030348       	|288            	|  8.7402         	|  29.708         	|
     get_train_batch                    	|  0.022253       	|288            	|  6.4087         	|  21.783         	|
     evaluation_step_and_end            	|  0.021021       	|277            	|  5.8228         	|  19.792         	|
     validation_step                    	|  0.020913       	|277            	|  5.7929         	|  19.69          	|
     training_step_and_backward         	|  0.019199       	|288            	|  5.5292         	|  18.794         	|
     model_forward                      	|  0.010312       	|288            	|  2.9697         	|  10.094         	|
     training_step                      	|  0.010076       	|288            	|  2.9017         	|  9.8631         	|
     model_backward                     	|  0.0076062      	|288            	|  2.1906         	|  7.4458         	|
     on_validation_epoch_end            	|  0.091247       	|2              	|  0.18249        	|  0.6203         	|
     on_train_batch_end                 	|  0.0003854      	|288            	|  0.111          	|  0.37728        	|
     on_train_start                     	|  0.09102        	|1              	|  0.09102        	|  0.30938        	|
     on_train_epoch_end                 	|  0.055033       	|1              	|  0.055033       	|  0.18706        	|
     cache_result                       	|  1.6399e-05     	|2290           	|  0.037554       	|  0.12765        	|
     on_validation_batch_end            	|  9.3069e-05     	|277            	|  0.02578        	|  0.087628       	|
     on_validation_end                  	|  0.008392       	|2              	|  0.016784       	|  0.05705        	|
     on_train_end                       	|  0.0075968      	|1              	|  0.0075968      	|  0.025822       	|
     on_batch_start                     	|  2.4213e-05     	|288            	|  0.0069733      	|  0.023702       	|
     on_after_backward                  	|  1.5198e-05     	|288            	|  0.004377       	|  0.014877       	|
     on_before_zero_grad                	|  1.4302e-05     	|288            	|  0.0041189      	|  0.014          	|
     on_train_batch_start               	|  1.264e-05      	|288            	|  0.0036403      	|  0.012374       	|
     on_batch_end                       	|  1.1853e-05     	|288            	|  0.0034136      	|  0.011603       	|
     training_step_end                  	|  9.123e-06      	|288            	|  0.0026274      	|  0.0089307      	|
     on_validation_batch_start          	|  8.5754e-06     	|277            	|  0.0023754      	|  0.008074       	|
     validation_step_end                	|  7.5117e-06     	|277            	|  0.0020807      	|  0.0070725      	|
     on_validation_start                	|  0.00019774     	|2              	|  0.00039547     	|  0.0013442      	|
     on_epoch_start                     	|  0.00021634     	|1              	|  0.00021634     	|  0.00073533     	|
     on_epoch_end                       	|  1.4326e-05     	|3              	|  4.2978e-05     	|  0.00014608     	|
     on_validation_epoch_start          	|  1.0761e-05     	|2              	|  2.1523e-05     	|  7.3157e-05     	|
     on_fit_start                       	|  2.0443e-05     	|1              	|  2.0443e-05     	|  6.9486e-05     	|
     on_train_epoch_start               	|  1.2379e-05     	|1              	|  1.2379e-05     	|  4.2077e-05     	|
     on_before_accelerator_backend_setup	|  1.1862e-05     	|1              	|  1.1862e-05     	|  4.0319e-05     	|

## Advanced Profiler

    hannah-train profiler=simple trainer.max_epochs=1 profiler=advanced

Creates python level performance summaries for the individual Training steps:


     Profile stats for: optimizer_step_and_closure_0
     8064 function calls in 8.897 seconds

     Ordered by: cumulative time
     List reduced from 25 to 20 due to restriction <20>

     ncalls  tottime  percall  cumtime  percall filename:lineno(function)
     288    0.001    0.000    8.897    0.031 accelerator.py:265(optimizer_step)
     288    0.000    0.000    8.896    0.031 accelerator.py:282(run_optimizer_step)
     288    0.000    0.000    8.896    0.031 training_type_plugin.py:159(optimizer_step)
     288    0.000    0.000    8.896    0.031 grad_mode.py:12(decorate_context)
     288    0.000    0.000    8.894    0.031 adamw.py:54(step)
     288    0.000    0.000    8.893    0.031 training_loop.py:648(train_step_and_backward_closure)
     288    0.000    0.000    8.893    0.031 training_loop.py:737(training_step_and_backward)
     288    0.000    0.000    8.892    0.031 contextlib.py:108(__enter__)
     288    0.000    0.000    8.892    0.031 {built-in method builtins.next}
     288    0.000    0.000    8.892    0.031 profilers.py:62(profile)
     288    0.000    0.000    8.892    0.031 profilers.py:250(start)
     288    8.892    0.031    8.892    0.031 {method 'enable' of '_lsprof.Profiler' objects}
     288    0.000    0.000    0.001    0.000 grad_mode.py:65(__enter__)
     288    0.000    0.000    0.000    0.000 accelerator.py:104(lightning_module)
     288    0.000    0.000    0.000    0.000 contextlib.py:238(helper)
     288    0.000    0.000    0.000    0.000 training_type_plugin.py:91(lightning_module)
     288    0.000    0.000    0.000    0.000 grad_mode.py:104(__enter__)
     288    0.000    0.000    0.000    0.000 contextlib.py:82(__init__)
     576    0.000    0.000    0.000    0.000 {built-in method torch._C.is_grad_enabled}
     288    0.000    0.000    0.000    0.000 base.py:82(unwrap_lightning_module)
