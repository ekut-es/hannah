#include <string>
#include <vector>

#include <dlib/dnn.h>

using namespace dlib;
using namespace std;

using anet_type = loss_multiclass_log <
  fc<10,
     fc<128,
	max_pool < 4, 1, 4, 1, 
		   con < 32, 5, 1, 1, 1,
			 max_pool < 4, 1, 4, 1, 
				    con < 16, 5, 1, 1, 1,
					  max_pool < 4, 1, 4, 1, 
						     con < 8, 5, 1, 1, 1,
							   input<matrix<float>>
							   > > >> >> >> >;


std::string global_path_sub = "Pfad Zu den Daten";



enum class Endianness {
  LittleEndian,
  BigEndian
};


int32_t fourBytesToInt(std::vector<uint8_t>& source, int startIndex, Endianness endianness = Endianness::LittleEndian) {
  int32_t result;

  if (endianness == Endianness::LittleEndian)
    result = (source[startIndex + 3] << 24) | (source[startIndex + 2] << 16) | (source[startIndex + 1] << 8) | source[startIndex];
  else
    result = (source[startIndex] << 24) | (source[startIndex + 1] << 16) | (source[startIndex + 2] << 8) | source[startIndex + 3];
  
  return result;
}


int16_t twoBytesToInt(std::vector<uint8_t>& source, int startIndex, Endianness endianness = Endianness::LittleEndian)
{
  int16_t result;
  
  if (endianness == Endianness::LittleEndian)
    result = (source[startIndex + 1] << 8) | source[startIndex];
  else
    result = (source[startIndex] << 8) | source[startIndex + 1];
  
  return result;
}




int getIndexOfString(std::vector<uint8_t>& source, std::string stringToSearchFor){
  int index = -1;
  int stringLength = (int)stringToSearchFor.length();
  
  for (int i = 0; i < source.size() - stringLength; i++)
    {
      std::string section(source.begin() + i, source.begin() + i + stringLength);
      
      if (section == stringToSearchFor)
	{
	  index = i;
	  break;
	}
    }
  
  return index;
}



float singleByteToSample(uint8_t sample) {
  return (static_cast<float> (sample) - 128.0f) / 128.0f;
}


float sixteenBitIntToSample(int16_t sample) {
  return (static_cast<float> (sample) - 32768.0f) / 32768.0f;
}




void load_alldata(std::vector<matrix<float>> &train_img, std::vector<unsigned long> &train_l, std::vector<matrix<float>> &test_img, std::vector<unsigned long> &test_l, std::string path) {

  train_img.clear();
  train_l.clear();
  test_img.clear();
  test_l.clear();
  
  std::string names[3];
  names[0] = "name1";
  names[1] = "name2";
  names[2] = "name3";
  


  for (int ii = 0; ii < 10; ii++) {
    for (int jj = 0; jj < 50; jj++) {
      for (int nn = 0; nn < 3; nn++) {
	
	
	
	matrix<float> data;
	
	
	std::string bas_path = global_path_sub + path + std::to_string(ii);
	bas_path += "_" + names[nn] + "_";
	bas_path += std::to_string(jj) + ".wav";
	
	std::ifstream file(bas_path, std::ios::binary);

	if (file.good()) {



	  file.unsetf(std::ios::skipws);
	  std::istream_iterator<uint8_t> begin(file), end;
	  std::vector<uint8_t> fileData(begin, end);



	  // HEADER CHUNK
	  std::string headerChunkID(fileData.begin(), fileData.begin() + 4);
	  //int32_t fileSizeInBytes = fourBytesToInt (fileData, 4) + 8;
	  std::string format(fileData.begin() + 8, fileData.begin() + 12);
	  
	  // -----------------------------------------------------------
	  // try and find the start points of key chunks
	  int indexOfDataChunk = getIndexOfString(fileData, "data");
	  int indexOfFormatChunk = getIndexOfString(fileData, "fmt");
	  
	  // if we can't find the data or format chunks, or the IDs/formats don't seem to be as expected
	  // then it is unlikely we'll able to read this file, so abort
	  if (indexOfDataChunk == -1 || indexOfFormatChunk == -1 || headerChunkID != "RIFF" || format != "WAVE"){
	    std::cout << "ERROR: this doesn't seem to be a valid .WAV file" << std::endl;
	  }
	  
	  
	  // -----------------------------------------------------------
	       // FORMAT CHUNK
	  int f = indexOfFormatChunk;
	  std::string formatChunkID(fileData.begin() + f, fileData.begin() + f + 4);
	  //int32_t formatChunkSize = fourBytesToInt (fileData, f + 4);
	  int16_t audioFormat = twoBytesToInt(fileData, f + 8);
	  int16_t numChannels = twoBytesToInt(fileData, f + 10);
	  uint32_t sampleRate = (uint32_t)fourBytesToInt(fileData, f + 12);
	  int32_t numBytesPerSecond = fourBytesToInt(fileData, f + 16);
	  int16_t numBytesPerBlock = twoBytesToInt(fileData, f + 20);
	  int bitDepth = (int)twoBytesToInt(fileData, f + 22);
	  
	  int numBytesPerSample = bitDepth / 8;
	  
	  // check that the audio format is PCM
	  if (audioFormat != 1)
	    {
	      std::cout << "ERROR: this is a compressed .WAV file and this library does not support decoding them at present" << std::endl;
	    }
	  
	  // check the number of channels is mono or stereo
	  if (numChannels < 1 || numChannels > 2)
	    {
	      std::cout << "ERROR: this WAV file seems to be neither mono nor stereo (perhaps multi-track, or corrupted?)" << std::endl;
	    }
	  
	  // check header data is consistent
	  if ((numBytesPerSecond != (numChannels * sampleRate * bitDepth) / 8) || (numBytesPerBlock != (numChannels * numBytesPerSample)))
	    {
	      std::cout << "ERROR: the header data in this WAV file seems to be inconsistent" << std::endl;
	    }
	  
	  // check bit depth is either 8, 16 or 24 bit
	  if (bitDepth != 8 && bitDepth != 16 && bitDepth != 24)
	    {
	      std::cout << "ERROR: this file has a bit depth that is not 8, 16 or 24 bits" << std::endl;
	    }
	  
	  // -----------------------------------------------------------
	  // DATA CHUNK
	  int d = indexOfDataChunk;
	  std::string dataChunkID(fileData.begin() + d, fileData.begin() + d + 4);
	  int32_t dataChunkSize = fourBytesToInt(fileData, d + 4);
	  
	  int numSamples = dataChunkSize / (numChannels * bitDepth / 8);
	  int samplesStartIndex = indexOfDataChunk + 8;
	  
	  
	  std::vector<std::vector<float>> samples;
	  samples.clear();
	  samples.resize(numChannels);
	  
	  for (int i = 0; i < numSamples; i++) {
	    for (int channel = 0; channel < numChannels; channel++) {
	      int sampleIndex = samplesStartIndex + (numBytesPerBlock * i) + channel * numBytesPerSample;
		  
	      if (bitDepth == 8){
		float sample = singleByteToSample(fileData[sampleIndex]);
		samples[channel].push_back(sample);
	      }
	      else if (bitDepth == 16){
		int16_t sampleAsInt = twoBytesToInt(fileData, sampleIndex);
		float sample = sixteenBitIntToSample(sampleAsInt);
		samples[channel].push_back(sample);
	      }
	      else if (bitDepth == 24) {
		int32_t sampleAsInt = 0;
		sampleAsInt = (fileData[sampleIndex + 2] << 16) | (fileData[sampleIndex + 1] << 8) | fileData[sampleIndex];
		      
		if (sampleAsInt & 0x800000) //  if the 24th bit is set, this is a negative number in 24-bit world
		  sampleAsInt = sampleAsInt | ~0xFFFFFF; // so make sure sign is extended to the 32 bit float
		
		float sample = (float)sampleAsInt / (float)8388608.;
		samples[channel].push_back(sample);
	      }
	      else{
		assert(false);
	      }
	    }
	  }



	  data.set_size(samples[0].size(), samples.size());

	  for (int i = 0; i < samples.size(); i++) {
	    for (int j = 0; j < samples[i].size(); j++) {
	      data(j, i) = samples[i][j];
	    }
	  }
	  
	  matrix<float> data_same_sz;
	  data_same_sz.set_size(2000, 1);//--------------2000
	  
	  resize_image(data, data_same_sz);
	  
	  if (jj < 40) {
	    train_img.push_back(data_same_sz);
	    train_l.push_back(ii);
	  }
	  else {
	    test_img.push_back(data_same_sz);
	    test_l.push_back(ii);
	  }
	  
	  
	}

      }
    }
  }
}


int main(int argc, char** argv) {
  try {
    std::vector<matrix<float>> train_img;
    std::vector<unsigned long> train_l;
  
    std::vector<matrix<float>> test_img;
    std::vector<unsigned long> test_l;
  
    

    int mini_batch_sz = 10;//100 oder 200
  
    load_alldata(train_img, train_l, test_img, test_l, "recordings/");
    
    std::cout << "Data size: " << train_img.size() << " | " << train_l.size() << std::endl;
    std::cout << "Data size: " << test_img.size() << " | " << test_l.size() << std::endl;
  
    std::cout << "Loading Done" << std::endl;
  
    anet_type net;

    dnn_trainer<anet_type, sgd> trainer(net, sgd(), { 0 });
    trainer.be_verbose();
    trainer.set_learning_rate(1e-2);
    trainer.set_mini_batch_size(mini_batch_sz);
  
    std::vector<matrix<float>> mini_batch_samples;
    std::vector<unsigned long> mini_batch_labels;
    dlib::rand rnd;
    
    int64 cnt = 0;
    while (cnt <  (train_l.size()/mini_batch_sz) * 100) {
      if (cnt < (train_l.size()/mini_batch_sz)) { //  1 epoche

	mini_batch_samples.clear();
	mini_batch_labels.clear();

	for (int sel_i = 0; sel_i < mini_batch_sz; sel_i++) {
	  int akv = rnd.get_integer(train_img.size());
	
	  mini_batch_samples.push_back(train_img[akv]);
	  mini_batch_labels.push_back(train_l[akv]);
	
	}

	trainer.train_one_step(mini_batch_samples.cbegin(), mini_batch_samples.cend(), mini_batch_labels.cbegin());

	cnt++;
      } else {
	cnt = 0;
      
	float richtig = 0;
	float all_cnt = 0;
      
	anet_type evnet = net;
	for (int i = 0; i < test_l.size(); i++) {
	
	  unsigned long dets = evnet.process(test_img[i]);
	  if (dets == test_l[i])
	    richtig++;
	  all_cnt++;
	}
	richtig /= all_cnt;
      
	std::cout << "Richtig: " << richtig  << std::endl;
      
      }
    }
  }catch (std::exception& e){
    cout << e.what() << endl;
  }
}
