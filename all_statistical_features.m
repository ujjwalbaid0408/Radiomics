% clc;
% clear all;
% close all;
%in=[2 5 1;9 5 2;7 8 5];
in = imread('cameraman.tif');

in = roi_flair;
in = double(in);

[row_new1, col_new1]= size(in);
cnt=zeros(1,256);
enery=0;sum=0;
voxels=abs(in);

for i=1:row_new1
    for j=1:col_new1
        enery=enery+(in(i,j)*in(i,j));       
        sum=sum+in(i,j);        
        for k=1:256            
            if in(i,j)==k
                cnt(k)=cnt(k)+1;
            end
        end
    end
end

 minimum=min(in(:));
 maximum=max(in(:));
 median_value=median(in(:));
 varience=var(in(:));
range=maximum-minimum;
display(range);

sum_of_intensities=sum;
display(sum_of_intensities);
display(enery);
display(minimum);
display(maximum);
display(median_value);

mean=sum/(row_new1*col_new1);
display(mean);

display(varience);

standard_dev=sqrt(varience);
display(standard_dev);

cnt=cnt/(row_new1*col_new1);
uniformity=cnt.*cnt;

entropyy=0;final_uniformity=0;

for i=1:256
    final_uniformity=final_uniformity+uniformity(i);
    if cnt(i)==0
        entropy=0;
    else
        entropy=-(cnt(i)*log2(cnt(i)));
    end
        entropyy=entropyy+entropy;
end

display(final_uniformity);
display(entropyy);