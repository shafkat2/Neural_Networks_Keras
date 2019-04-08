import pywt


#discrete wavelet analysis for 1S signal
def denoise(data,wavelet,levels,level,alpha):
    WC = pywt.wavedec(data,wavelet,mode = "per",level = level)
    noiseSigma = mad( WC[-levels] )
    threshold=alpha*noiseSigma*math.sqrt(math.log2(data.size))
    NWC = map(lambda x: pywt.threshold(x,threshold,'garotte'), WC)
    NWC = list(NWC)
    NWC = pywt.waverec( NWC, wavelet,mode= "per")
    NWC = pd.DataFrame(NWC)
    return NWC

def wavlet_analysis(Data,batch_size,wavelet,levels,level,alpha): 
  #wavelet analysis
  batch_s = batch_size
  ward_list = pd.DataFrame()
  size1 = 0
  size2 =  batch_s     
  count_row_test = Data.shape[0]
  num_batches =int(round(count_row_test/ batch_s)) 
  for x in range(num_batches):
    
    batch2 = Data.iloc[size1:size2]
    batch_noise = batch2.values.squeeze()
    ward = denoise(batch_noise,wavelet,levels,level,alpha)
    ward_list = ward_list.append(ward,ignore_index=True)
    size1 = size1+ batch_s
    size2 = size2+ batch_s
  ward_list = ward_list.values
  #interpolation
  ward_list = fill_nans_scipy1(ward_list)
  ward_list = pd.DataFrame(ward_list)
  plt.plot(ward_list)
  return ward_list

# mini batch wavlet analysis
  def Mini_batch_wavelet_denoising(data,batch,batch_tweak,wavelet,level_to_denoise,level_of_localization,alpha):
  x1 = 0
  x2 = batch
  denoised = pd.DataFrame()
  for x in range(int(data.shape[0]/batch)):

    T  = wavlet_analysis(data[0].iloc[x1:x2],int(data[x1:x2].shape[0]/batch_tweak),wavelet,level_to_denoise,level_of_localization,alpha)
    denoised = denoised.append(T,ignore_index= True)
    x1 = x1 + batch
    x2 = x2 + batch

  print(denoised.shape)
  plt.figure(figsize = (18,8))
  plt.plot(denoised)
  return denoised
  