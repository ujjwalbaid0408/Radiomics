% This program is developed at Center of Excellence, SGGSIET, Nanded 
% Author: Ujjwal Baid

%%

clc;
clear all;
close all;

tic

% pathname = 'D:\DNN_PatchGeneration';
pathname = 'D:\BRATS\BRATS2018\Validation_All_in_one';

extension_gt = '.nii';
extension_t1ce = '_t1ce.nii';

% data = importdata('BRATS2017List.xlsx');
input_data = importdata('survival_evaluation.csv');

[row_data1, col_data1, dim_data] = size(input_data.data);

%output_data = 'For slice with Maximum region count.xlsx';
output_data = 'Radiomic Features_Validation_Edema+Enhance+Core_t1ce.xlsx';

sheet = 1;
data1 = {'Patient Name', 'Edema Count', 'Edema Slice' 'Core Count', 'Core Slice', 'Enhanc Count', 'Enhanc Slice',...
           'CoveredIntensityRange', 'Energy', 'Entropy', 'Maximum', 'Mean', 'Minimum', 'Range', 'Median', 'Variance', ...
           'SumOfIntensities', 'StandardDeviation', 'Uniformity', 'autoCorrelation', 'clusterProminence',...
           'clusterShade', 'contrast', 'correlation', 'differenceEntropy', 'differenceVariance', 'dissimilarity',...
           'energy', 'entropy', 'homogeneity', 'informationMeasureOfCorrelation1', 'informationMeasureOfCorrelation2',...
           'inverseDifference', 'maximumProbability', 'sumAverage', 'sumEntropy', 'sumOfSquaresVariance', 'sumVariance'... 
           ... % List of Approximation coefficient of wavelet transform
           'ca_autoCorrelation', 'ca_clusterProminence', 'ca_clusterShade',  'ca_contrast', 'ca_correlation', ...
           'ca_differenceEntropy', 'ca_differenceVariance','ca_dissimilarity', 'ca_energy', 'ca_entropy',...
           'ca_homogeneity', 'ca_informationMeasureOfCorrelation1', 'ca_informationMeasureOfCorrelation2',...
           'ca_inverseDifference', 'ca_maximumProbability', 'ca_sumAverage', 'ca_sumEntropy','ca_sumOfSquaresVariance',...
           'ca_sumVariance'...
            ... % List of Coefficients in Horizontal direction of wavelet transform
           'chd_autoCorrelation', 'chd_clusterProminence', 'chd_clusterShade',  'chd_contrast', 'chd_correlation', ...
           'chd_differenceEntropy', 'chd_differenceVariance','chd_dissimilarity', 'chd_energy', 'chd_entropy',...
           'chd_homogeneity', 'chd_informationMeasureOfCorrelation1', 'chd_informationMeasureOfCorrelation2',...
           'chd_inverseDifference', 'chd_maximumProbability', 'chd_sumAverage', 'chd_sumEntropy',...
           'chd_sumOfSquaresVariance', 'chd_sumVariance'...
            ... % List of Coefficients in Vertical direction of wavelet transform
           'cvd_autoCorrelation', 'cvd_clusterProminence', 'cvd_clusterShade',  'cvd_contrast', 'cvd_correlation',...
           'cvd_differenceEntropy', 'cvd_differenceVariance','cvd_dissimilarity', 'cvd_energy', 'cvd_entropy', ...
           'cvd_homogeneity', 'cvd_informationMeasureOfCorrelation1', 'cvd_informationMeasureOfCorrelation2',...
           'cvd_inverseDifference', 'cvd_maximumProbability', 'cvd_sumAverage', 'cvd_sumEntropy',...
           'cvd_sumOfSquaresVariance', 'cvd_sumVariance'...
           ... % List of Coefficients in Diagonal direction of wavelet transform
           'cdd_autoCorrelation', 'cdd_clusterProminence', 'cdd_clusterShade',  'cdd_contrast', 'cdd_correlation',...
           'cdd_differenceEntropy', 'cdd_differenceVariance','cdd_dissimilarity', 'cdd_energy', 'cdd_entropy', ...
           'cdd_homogeneity', 'cdd_informationMeasureOfCorrelation1', 'cdd_informationMeasureOfCorrelation2',...
           'cdd_inverseDifference', 'cdd_maximumProbability', 'cdd_sumAverage', 'cdd_sumEntropy',...
           'cdd_sumOfSquaresVariance', 'cdd_sumVariance'};

       
    xlRange_initial = 'A1';
    xlswrite(output_data,data1,sheet,xlRange_initial)

    xlrange = 'A';

%% Patch Extraction Framework

for row_data=2:row_data1+1 % 2 is replces by 6
%for row_data=1
    
    filename_GT1 = input_data.textdata{row_data,1};       filename_GT = strcat(filename_GT1,extension_gt);
    info_GT = nii_read_header(filename_GT, pathname);       V_GT = nii_read_volume(info_GT);   
    
    
    filename_t1ce = input_data.textdata{row_data,1};       filename_t1ce = strcat(filename_t1ce,extension_t1ce);
    info_t1ce = nii_read_header(filename_t1ce, pathname);    V_t1ce = nii_read_volume(info_t1ce);   
    
    
    [row, col, dim]=size(V_GT);
    % 1 for NCR and NET, 2 for ED, 4 for ET, and 0 for everything else
    sprintf('Patient number = %d and ID = %s',row_data,filename_GT)
   
    max_Core = 0;
    max_Edema = 0;
    max_Enhan = 0;
    
    max_Core_slice = 1;
    max_Edema_slice = 1;
    max_Enhan_slice = 1;
    
    
    for d=1: dim
        
        Counter_Core = 0;
        Counter_Edema = 0;
        Counter_Enhan = 0;   
        % sprintf('Slice = %d',d)                
        img_gt = V_GT(:,:,d);         img_gt=img_gt';   
        
        [row, col]=size(img_gt);        
        for r=1:row        
            for c=1:col
                            
                if img_gt(r,c) == 1
                Counter_Core = Counter_Core + 1;                                
                end
                
                if img_gt(r,c) == 2
                Counter_Edema = Counter_Edema + 1;                                
                end
                
                if img_gt(r,c) == 4
                Counter_Enhan = Counter_Enhan + 1;                                
                end                
            end
        end   
        
        if Counter_Core > max_Core           
            max_Core = Counter_Core;
            max_Core_slice = d;
        end
        if  Counter_Edema > max_Edema           
            max_Edema = Counter_Edema;
            max_Edema_slice = d;
        end
        if Counter_Enhan > max_Enhan
            max_Enhan = Counter_Enhan;
            max_Enhan_slice = d;
        end
        
        
    end
    
   
    
    in_bw=V_GT(:,:,max_Enhan_slice);
    CC = bwconncomp(in_bw);
    l = [];
    for i = 1:length(CC.PixelIdxList) % For each spot
        %[r, c] = ind2sub(size(in_bw),CC.PixelIdxList{i}); % Calculate indices
         l(i) = length(CC.PixelIdxList{i}); % Measure no. of pixels
    end

    [Maximum index] = max(l)
    [r, c] = ind2sub(size(in_bw),CC.PixelIdxList{index}); % Calculate indices
    
    row_min = min(r);
    row_max = max(r);
    col_min = min(c);
    col_max = max(c);

    roi_t1ce = V_t1ce(row_min:row_max, col_min:col_max, max_Enhan_slice);
    roi_t1ce = double(roi_t1ce);
    image_t1ce = V_t1ce(:,:,max_Enhan_slice);
    
%     figure, imshow(roi_t1ce,[]);
%     figure, imshow(in_bw,[]); 
     
   %% Calculate all statistical features
   
    roi_min = min(roi_t1ce(:));
    roi_max = max(roi_t1ce(:));
    image_min = min(image_t1ce(:));
    image_max = max(image_t1ce(:));
   
    CoveredIntensityRange = (roi_max - roi_min)/(image_max - image_min); 
    Energy = sum(sum(roi_t1ce.*roi_t1ce));
   
   
    in = roi_t1ce;
    in = double(in);

    [row_new1, col_new1]= size(in);
    cnt=zeros(1,256);
    Energy=0;sum1=0;
    voxels=abs(in);

    for i=1:row_new1
        for j=1:col_new1
            Energy=Energy+(in(i,j)*in(i,j));       
            sum1=sum1+in(i,j);        
            for k=1:256            
                if in(i,j)==k
                    cnt(k)=cnt(k)+1;
                end
            end
        end
    end
    
    Mean=sum1/(row_new1*col_new1);
    Minimum=min(in(:));
    Maximum=max(in(:));
    Range = Maximum - Minimum;
    Median=median(in(:));
    Variance=var(in(:));
    SumOfIntensities=sum1;
    StandardDeviation=sqrt(Variance);
        
    cnt=cnt/(row_new1*col_new1);
    uniformity_temp=cnt.*cnt;

    Entropy=0;Uniformity=0;

    for i=1:256
        Uniformity=Uniformity+uniformity_temp(i);
        if cnt(i)==0
            entropy=0;
        else
            entropy=-(cnt(i)*log2(cnt(i)));
        end
            Entropy=Entropy+entropy;
    end

%% Calculate GLCM features
    
    % GLCM of original image
    glcm_t1ce = graycomatrix(roi_t1ce,'NumLevels',9,'Offset',[0 1],'GrayLimits',[]);
    GLCMFeatures_t1ce = GLCMFeatures(glcm_t1ce);
    
    % For GLCM of Undecimated Wavelet Transform
    [row_roi, col_roi] = size(roi_t1ce);
     m_row = mod(row_roi,2)
     m_col = mod(col_roi,2)
     
    if m_row == 1
     roi_t1ce =   imresize(roi_t1ce,[ row_roi+1,  col_roi]);       
    end    
    if m_col == 1
        roi_t1ce = imresize(roi_t1ce,[row_roi, col_roi+1]);     
    end    
    if m_row ==1 && m_col == 1
       roi_t1ce =  imresize(roi_t1ce,[row_roi+1, col_roi+1]);        
    end
       
   [ca,chd,cvd,cdd] = swt2(roi_t1ce,1,'sym4'); 
   
   glcm_t1ce_ca = graycomatrix(roi_t1ce,'NumLevels',9,'Offset',[0 1],'GrayLimits',[]);
   GLCMFeatures_t1ce_ca = GLCMFeatures(glcm_t1ce_ca);  
   
   glcm_t1ce_chd = graycomatrix(roi_t1ce,'NumLevels',9,'Offset',[0 1],'GrayLimits',[]);
   GLCMFeatures_t1ce_chd = GLCMFeatures(glcm_t1ce_chd);  
   
   glcm_t1ce_cvd = graycomatrix(roi_t1ce,'NumLevels',9,'Offset',[0 1],'GrayLimits',[]);
   GLCMFeatures_t1ce_cvd = GLCMFeatures(glcm_t1ce_cvd);  
   
   glcm_t1ce_cdd = graycomatrix(roi_t1ce,'NumLevels',9,'Offset',[0 1],'GrayLimits',[]);
   GLCMFeatures_t1ce_cdd = GLCMFeatures(glcm_t1ce_cdd);  
   
   
 %% GLRLM Features
 
   GLRLMS = grayrlmatrix(roi_t1ce,'Offset', 1, 'NumLevels',8,'GrayLimits',[]);  
   stats = grayrlprops(GLRLMS);
   
%%     
           
       
     
    p = int2str(row_data);
    cell_address = strcat(xlrange,p);
    data2 = {filename_GT1,max_Edema,max_Edema_slice, max_Core, max_Core_slice, max_Enhan, max_Enhan_slice, ...
             CoveredIntensityRange, Energy, Entropy, Maximum, Mean, Minimum, Range, Median, Variance, ...
             SumOfIntensities, StandardDeviation, Uniformity, GLCMFeatures_t1ce.autoCorrelation,...
             GLCMFeatures_t1ce.clusterProminence,  GLCMFeatures_t1ce.clusterShade,  GLCMFeatures_t1ce.contrast,...
             GLCMFeatures_t1ce.correlation, GLCMFeatures_t1ce.differenceEntropy, GLCMFeatures_t1ce.differenceVariance,...
             GLCMFeatures_t1ce.dissimilarity, GLCMFeatures_t1ce.energy, GLCMFeatures_t1ce.entropy,...
             GLCMFeatures_t1ce.homogeneity, GLCMFeatures_t1ce.informationMeasureOfCorrelation1, ...
             GLCMFeatures_t1ce.informationMeasureOfCorrelation2,  GLCMFeatures_t1ce.inverseDifference,...
             GLCMFeatures_t1ce.maximumProbability, GLCMFeatures_t1ce.sumAverage, GLCMFeatures_t1ce.sumEntropy, ...
             GLCMFeatures_t1ce.sumOfSquaresVariance, GLCMFeatures_t1ce.sumVariance,...
             ... ... % List of Approximation coefficient of wavelet transform
             GLCMFeatures_t1ce_ca.autoCorrelation, ...
             GLCMFeatures_t1ce_ca.clusterProminence,  GLCMFeatures_t1ce_ca.clusterShade,  GLCMFeatures_t1ce_ca.contrast,...
             GLCMFeatures_t1ce_ca.correlation, GLCMFeatures_t1ce_ca.differenceEntropy, GLCMFeatures_t1ce_ca.differenceVariance,...
             GLCMFeatures_t1ce_ca.dissimilarity, GLCMFeatures_t1ce_ca.energy, GLCMFeatures_t1ce_ca.entropy,...
             GLCMFeatures_t1ce_ca.homogeneity, GLCMFeatures_t1ce_ca.informationMeasureOfCorrelation1, ...
             GLCMFeatures_t1ce_ca.informationMeasureOfCorrelation2,  GLCMFeatures_t1ce_ca.inverseDifference,...
             GLCMFeatures_t1ce_ca.maximumProbability, GLCMFeatures_t1ce_ca.sumAverage, GLCMFeatures_t1ce_ca.sumEntropy, ...
             GLCMFeatures_t1ce_ca.sumOfSquaresVariance, GLCMFeatures_t1ce_ca.sumVariance...
             ... ... % List of Coefficient in Horizontal direction of wavelet transform             
             GLCMFeatures_t1ce_chd.autoCorrelation, ...
             GLCMFeatures_t1ce_chd.clusterProminence,  GLCMFeatures_t1ce_chd.clusterShade,  GLCMFeatures_t1ce_chd.contrast,...
             GLCMFeatures_t1ce_chd.correlation, GLCMFeatures_t1ce_chd.differenceEntropy, GLCMFeatures_t1ce_chd.differenceVariance,...
             GLCMFeatures_t1ce_chd.dissimilarity, GLCMFeatures_t1ce_chd.energy, GLCMFeatures_t1ce_chd.entropy,...
             GLCMFeatures_t1ce_chd.homogeneity, GLCMFeatures_t1ce_chd.informationMeasureOfCorrelation1, ...
             GLCMFeatures_t1ce_chd.informationMeasureOfCorrelation2,  GLCMFeatures_t1ce_chd.inverseDifference,...
             GLCMFeatures_t1ce_chd.maximumProbability, GLCMFeatures_t1ce_chd.sumAverage, GLCMFeatures_t1ce_chd.sumEntropy, ...
             GLCMFeatures_t1ce_chd.sumOfSquaresVariance, GLCMFeatures_t1ce_chd.sumVariance...
             ... ... % List of Coefficient in Vertical direction of wavelet transform             
             GLCMFeatures_t1ce_cvd.autoCorrelation, ...
             GLCMFeatures_t1ce_cvd.clusterProminence,  GLCMFeatures_t1ce_cvd.clusterShade,  GLCMFeatures_t1ce_cvd.contrast,...
             GLCMFeatures_t1ce_cvd.correlation, GLCMFeatures_t1ce_cvd.differenceEntropy, GLCMFeatures_t1ce_cvd.differenceVariance,...
             GLCMFeatures_t1ce_cvd.dissimilarity, GLCMFeatures_t1ce_cvd.energy, GLCMFeatures_t1ce_cvd.entropy,...
             GLCMFeatures_t1ce_cvd.homogeneity, GLCMFeatures_t1ce_cvd.informationMeasureOfCorrelation1, ...
             GLCMFeatures_t1ce_cvd.informationMeasureOfCorrelation2,  GLCMFeatures_t1ce_cvd.inverseDifference,...
             GLCMFeatures_t1ce_cvd.maximumProbability, GLCMFeatures_t1ce_cvd.sumAverage, GLCMFeatures_t1ce_cvd.sumEntropy, ...
             GLCMFeatures_t1ce_cvd.sumOfSquaresVariance, GLCMFeatures_t1ce_cvd.sumVariance...
             ... ... % List of Coefficient in Diagonal direction of wavelet transform 
             GLCMFeatures_t1ce_cdd.autoCorrelation, ...
             GLCMFeatures_t1ce_cdd.clusterProminence,  GLCMFeatures_t1ce_cdd.clusterShade,  GLCMFeatures_t1ce_cdd.contrast,...
             GLCMFeatures_t1ce_cdd.correlation, GLCMFeatures_t1ce_cdd.differenceEntropy, GLCMFeatures_t1ce_cdd.differenceVariance,...
             GLCMFeatures_t1ce_cdd.dissimilarity, GLCMFeatures_t1ce_cdd.energy, GLCMFeatures_t1ce_cdd.entropy,...
             GLCMFeatures_t1ce_cdd.homogeneity, GLCMFeatures_t1ce_cdd.informationMeasureOfCorrelation1, ...
             GLCMFeatures_t1ce_cdd.informationMeasureOfCorrelation2,  GLCMFeatures_t1ce_cdd.inverseDifference,...
             GLCMFeatures_t1ce_cdd.maximumProbability, GLCMFeatures_t1ce_cdd.sumAverage, GLCMFeatures_t1ce_cdd.sumEntropy, ...
             GLCMFeatures_t1ce_cdd.sumOfSquaresVariance, GLCMFeatures_t1ce_cdd.sumVariance};        
        
    xlswrite(output_data,data2,sheet,cell_address);
    
         
     close all;    
   toc
end
