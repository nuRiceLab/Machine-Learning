// running inference in C++ 

// necessary handlers 
#include "tensorflow/cc/saved_model/loader.h"
#include "tensorflow/core/framework/tensor.h"

// load the saved model 
using namespace tensorflow;

const std::string model_dir = "/path/to/saved/cvn_atmnu"; // TODO: CHANGE!!! 
const std::string tags = {tensorflow::kSavedModelTagServe}; // The tag used at saving time. This is standard but not sure. 
SavedModelBundleLite bundle;
SessionOptions session_options;
RunOptions run_options;

Status status = LoadSavedModel(session_options, run_options, model_dir, {tags}, &bundle);
if (!status.ok()) {
    std::cerr << "Failed to load saved model: " << status.ToString() << std::endl;
    return 1;
}

// prepare input tensors (currently saying batch size is 1 because idk what the workflow will do) 
Tensor input_tensor(DT_FLOAT, TensorShape({1, 200, 200, 3}));

// TODO: Populate your tensor with image data here before running the model

// Init the output object
std::vector<Tensor> outputs;

// run the model 
status = bundle.GetSession()->Run({{"input_1", input_tensor}},
                                  {"flavour", "protons", "pions"}, // Output operation names
                                  {}, &outputs);

if (!status.ok()) {
    std::cerr << "Failed to run model: " << status.ToString() << std::endl;
    return 1;
}

// Assuming the status is OK and the model ran successfully, access the outputs
// The outputs vector should now contain three tensors corresponding to the specified names
if (outputs.size() == 3) {
    const Tensor& flavour_output = outputs[0]; // Access the 'flavour' output
    const Tensor& protons_output = outputs[1]; // Access the 'protons' output
    const Tensor& pions_output = outputs[2];   // Access the 'pions' output

    // TODO: Now process these tensors as needed
    // Each line is a placeholder for our eventual inference. 
    std::cout << "Flavour output shape: " << flavour_output.shape().DebugString() << std::endl;
    std::cout << "Protons output shape: " << protons_output.shape().DebugString() << std::endl;
    std::cout << "Pions output shape: " << pions_output.shape().DebugString() << std::endl;
}
