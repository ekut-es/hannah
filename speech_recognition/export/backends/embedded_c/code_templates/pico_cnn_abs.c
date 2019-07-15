for(int i = 0; i < {{input_buffer.typed_size}}; i++){
  {{output_buffer.start_ptr}}[i] = fabsf({{input_buffer.start_ptr}}[i]);
}
