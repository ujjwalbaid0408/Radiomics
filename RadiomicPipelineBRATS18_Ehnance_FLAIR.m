% This program is developed at Center of Excellence, SGGSIET, Nanded 
% Author: Ujjwal Baid


% In program is for Radiomic Features on Enh. tumor part on Flair
% Considering max enhancing slice
%%

clc;
clear all;
close all;

tic

% pathname = 'D:\DNN_PatchGeneration';
pathname = 'D:\BRATS\BRATS2018\Validation_All_in_one';

extension_gt = '.nii';
extension_flair = '_flair.nii';

% data = importdata('BRATS2017List.xlsx');
input_data = importdata('survival_evaluation.csv');

[row_data1, col_data1, dim_data] = size(input_data.data);

%output_data = 'For slice with Maximum region count.xlsx';
output_data = 'Radiomic Features_Validation_Enhance_flair.xlsx';

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

for row_data=2:row_data1+1
%for row_data=1
    
    filename_GT1 = input_data.textdata{row_data,1};       filename_GT = strcat(filename_GT1,extension_gt);
    info_GT = nii_read_header(filename_GT, pathname);       V_GT = nii_read_volume(info_GT);   
    
    
    filename_flair = input_data.textdata{row_data,1};       filename_flair = strcat(filename_flair,extension_flair);
    info_flair = nii_read_header(filename_flair, pathname);    V_flair = nii_read_volume(info_flair);   
    
    
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
    in_bw1 = in_bw;
    [row_bw, col_bw] = size(in_bw);
    
    for x = 1:row_bw
        for y = 1:col_bw
            if in_bw(x,y) ==4
                in_bw(x,y)=1;
            else
                in_bw(x,y)=0;
            end
        end
    end
    
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

    temp1 = in_bw(row_min:row_max, col_min:col_max);
    temp1 = int16(temp1);
    temp2 = V_flair(row_min:row_max, col_min:col_max, max_Enhan_slice);
    
    temp3 = temp1.*temp2;
    roi_flair = temp3;
    
    roi_flair = double(roi_flair);
    image_flair = V_flair(:,:,max_Enhan_slice);
    
%     figure, imshow(roi_flair,[]);
%     figure, imshow(in_bw,[]); 
     
   %% Calculate all statistical features
   
    roi_min = min(roi_flair(:));
    roi_max = max(roi_flair(:));
    image_min = min(image_flair(:));
    image_max = max(image_flair(:));
   
    CoveredIntensityRange = (roi_max - roi_min)/(image_max - image_min); 
    Energy = sum(sum(roi_flair.*roi_flair));
   
   
    in = roi_flair;
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
    glcm_flair = graycomatrix(roi_flair,'NumLevels',9,'Offset',[0 1],'GrayLimits',[]);
    GLCMFeatures_flair = GLCMFeatures(glcm_flair);
    
    % For GLCM of Undecimated Wavelet Transform
    [row_roi, col_roi] = size(roi_flair);
     m_row = mod(row_roi,2)
     m_col = mod(col_roi,2)
     
    if m_row == 1
     roi_flair =   imresize(roi_flair,[ row_roi+1,  col_roi]);       
    end    
    if m_col == 1
        roi_flair = imresize(roi_flair,[row_roi, col_roi+1]);     
    end    
    if m_row ==1 && m_col == 1
       roi_flair =  imresize(roi_flair,[row_roi+1, col_roi+1]);        
    end
       
   [ca,chd,cvd,cdd] = swt2(roi_flair,1,'sym4'); 
   
   glcm_flair_ca = graycomatrix(roi_flair,'NumLevels',9,'Offset',[0 1],'GrayLimits',[]);
   GLCMFeatures_flair_ca = GLCMFeatures(glcm_flair_ca);  
   
   glcm_flair_chd = graycomatrix(roi_flair,'NumLevels',9,'Offset',[0 1],'GrayLimits',[]);
   GLCMFeatures_flair_chd = GLCMFeatures(glcm_flair_chd);  
   
   glcm_flair_cvd = graycomatrix(roi_flair,'NumLevels',9,'Offset',[0 1],'GrayLimits',[]);
   GLCMFeatures_flair_cvd = GLCMFeatures(glcm_flair_cvd);  
   
   glcm_flair_cdd = graycomatrix(roi_flair,'NumLevels',9,'Offset',[0 1],'GrayLimits',[]);
   GLCMFeatures_flair_cdd = GLCMFeatures(glcm_flair_cdd);  
   
   
 %% GLRLM Features
 
   GLRLMS = grayrlmatrix(roi_flair,'Offset', 1, 'NumLevels',8,'GrayLimits',[]);  
   stats = grayrlprops(GLRLMS);
   
%%     
           
       
     
    p = int2str(row_data);
    cell_address = strcat(xlrange,p);
    data2 = {filename_GT1,max_Edema,max_Edema_slice, max_Core, max_Core_slice, max_Enhan, max_Enhan_slice, ...
             CoveredIntensityRange, Energy, Entropy, Maximum, Mean, Minimum, Range, Median, Variance, ...
             SumOfIntensities, StandardDeviation, Uniformity, GLCMFeatures_flair.autoCorrelation,...
             GLCMFeatures_flair.clusterProminence,  GLCMFeatures_flair.clusterShade,  GLCMFeatures_flair.contrast,...
             GLCMFeatures_flair.correlation, GLCMFeatures_flair.differenceEntropy, GLCMFeatures_flair.differenceVariance,...
             GLCMFeatures_flair.dissimilarity, GLCMFeatures_flair.energy, GLCMFeatures_flair.entropy,...
             GLCMFeatures_flair.homogeneity, GLCMFeatures_flair.informationMeasureOfCorrelation1, ...
             GLCMFeatures_flair.informationMeasureOfCorrelation2,  GLCMFeatures_flair.inverseDifference,...
             GLCMFeatures_flair.maximumProbability, GLCMFeatures_flair.sumAverage, GLCMFeatures_flair.sumEntropy, ...
             GLCMFeatures_flair.sumOfSquaresVariance, GLCMFeatures_flair.sumVariance,...
             ... ... % List of Approximation coefficient of wavelet transform
             GLCMFeatures_flair_ca.autoCorrelation, ...
             GLCMFeatures_flair_ca.clusterProminence,  GLCMFeatures_flair_ca.clusterShade,  GLCMFeatures_flair_ca.contrast,...
             GLCMFeatures_flair_ca.correlation, GLCMFeatures_flair_ca.differenceEntropy, GLCMFeatures_flair_ca.differenceVariance,...
             GLCMFeatures_flair_ca.dissimilarity, GLCMFeatures_flair_ca.energy, GLCMFeatures_flair_ca.entropy,...
             GLCMFeatures_flair_ca.homogeneity, GLCMFeatures_flair_ca.informationMeasureOfCorrelation1, ...
             GLCMFeatures_flair_ca.informationMeasureOfCorrelation2,  GLCMFeatures_flair_ca.inverseDifference,...
             GLCMFeatures_flair_ca.maximumProbability, GLCMFeatures_flair_ca.sumAverage, GLCMFeatures_flair_ca.sumEntropy, ...
             GLCMFeatures_flair_ca.sumOfSquaresVariance, GLCMFeatures_flair_ca.sumVariance...
             ... ... % List of Coefficient in Horizontal direction of wavelet transform             
             GLCMFeatures_flair_chd.autoCorrelation, ...
             GLCMFeatures_flair_chd.clusterProminence,  GLCMFeatures_flair_chd.clusterShade,  GLCMFeatures_flair_chd.contrast,...
             GLCMFeatures_flair_chd.correlation, GLCMFeatures_flair_chd.differenceEntropy, GLCMFeatures_flair_chd.differenceVariance,...
             GLCMFeatures_flair_chd.dissimilarity, GLCMFeatures_flair_chd.energy, GLCMFeatures_flair_chd.entropy,...
             GLCMFeatures_flair_chd.homogeneity, GLCMFeatures_flair_chd.informationMeasureOfCorrelation1, ...
             GLCMFeatures_flair_chd.informationMeasureOfCorrelation2,  GLCMFeatures_flair_chd.inverseDifference,...
             GLCMFeatures_flair_chd.maximumProbability, GLCMFeatures_flair_chd.sumAverage, GLCMFeatures_flair_chd.sumEntropy, ...
             GLCMFeatures_flair_chd.sumOfSquaresVariance, GLCMFeatures_flair_chd.sumVariance...
             ... ... % List of Coefficient in Vertical direction of wavelet transform             
             GLCMFeatures_flair_cvd.autoCorrelation, ...
             GLCMFeatures_flair_cvd.clusterProminence,  GLCMFeatures_flair_cvd.clusterShade,  GLCMFeatures_flair_cvd.contrast,...
             GLCMFeatures_flair_cvd.correlation, GLCMFeatures_flair_cvd.differenceEntropy, GLCMFeatures_flair_cvd.differenceVariance,...
             GLCMFeatures_flair_cvd.dissimilarity, GLCMFeatures_flair_cvd.energy, GLCMFeatures_flair_cvd.entropy,...
             GLCMFeatures_flair_cvd.homogeneity, GLCMFeatures_flair_cvd.informationMeasureOfCorrelation1, ...
             GLCMFeatures_flair_cvd.informationMeasureOfCorrelation2,  GLCMFeatures_flair_cvd.inverseDifference,...
             GLCMFeatures_flair_cvd.maximumProbability, GLCMFeatures_flair_cvd.sumAverage, GLCMFeatures_flair_cvd.sumEntropy, ...
             GLCMFeatures_flair_cvd.sumOfSquaresVariance, GLCMFeatures_flair_cvd.sumVariance...
             ... ... % List of Coefficient in Diagonal direction of wavelet transform 
             GLCMFeatures_flair_cdd.autoCorrelation, ...
             GLCMFeatures_flair_cdd.clusterProminence,  GLCMFeatures_flair_cdd.clusterShade,  GLCMFeatures_flair_cdd.contrast,...
             GLCMFeatures_flair_cdd.correlation, GLCMFeatures_flair_cdd.differenceEntropy, GLCMFeatures_flair_cdd.differenceVariance,...
             GLCMFeatures_flair_cdd.dissimilarity, GLCMFeatures_flair_cdd.energy, GLCMFeatures_flair_cdd.entropy,...
             GLCMFeatures_flair_cdd.homogeneity, GLCMFeatures_flair_cdd.informationMeasureOfCorrelation1, ...
             GLCMFeatures_flair_cdd.informationMeasureOfCorrelation2,  GLCMFeatures_flair_cdd.inverseDifference,...
             GLCMFeatures_flair_cdd.maximumProbability, GLCMFeatures_flair_cdd.sumAverage, GLCMFeatures_flair_cdd.sumEntropy, ...
             GLCMFeatures_flair_cdd.sumOfSquaresVariance, GLCMFeatures_flair_cdd.sumVariance};        
        
    xlswrite(output_data,data2,sheet,cell_address);
    
         
     close all;    
   toc
end
