% clc;
% clear all;
% close all; 

%in=[2 5 1;9 5 2;7 8 5];
in = imread('cameraman.tif');
%in = crop; 

[row col]=size(in);

[glcm, SI] = graycomatrix(in,'NumLevels',10,'GrayLimits',[])

glcm_mean=mean(glcm(:));
fprintf('\n glcm matrix mean is %f \n',glcm_mean);

glcm_std=std(glcm(:));
fprintf('\n glcm matrix standard deviation is %f \n',glcm_std);

[m n]=size(glcm);
cnt=0;

for k=1:256;
    for i=1:m;
        for j=1:n;
            if glcm(i,j)==(k-1)
                cnt=cnt+1;
            end
        end
    end
    array(k)=cnt;
    cnt=0;
end

Ng=m;

autocorr=0;cluster_prominance=0;cluster_shade=0;cluster_tendancy=0;contrast=0;
correlationn=0;dissimilarity=0;energyy=0;entropyy=0;
Px(i)=0;harralick_corr=0;

for k=1:256
    for i=1:Ng
        for j=1:Ng
            if glcm(i,j)==(k-1)
               glcm(i,j)=(array(k)./(Ng.*Ng));
            end
            Px(i)=Px(i)+glcm(i,j);
        end
    end
end

fprintf('\n glcm matrix probability is %f \n',Px(i));

Mx=mean(Px);
fprintf('\n mean of prob of glcm is %f \n',Mx);

Std_x=std(Px(:));
fprintf('\n standard deviation of prob of glcm is %f \n',glcm_std);

for k=1:256
    for i=1:Ng
        for j=1:Ng
            if glcm(i,j)==(k-1)
               glcm(i,j)=(glcm(i,j)./(Ng.*Ng));
            end
                autocorr=autocorr+(i.*j.*(glcm(i,j)));
                cluster_prominance=cluster_prominance+(((i+j-(2.*glcm_mean)).^4).*(glcm(i,j)));
                cluster_shade=cluster_shade+(((i+j-(2.*glcm_mean)).^3).*(glcm(i,j)));
                cluster_tendancy=cluster_tendancy+(((i+j-(2.*glcm_mean)).^2).*(glcm(i,j)));
                contrast=contrast+(((i-j).^2).*(glcm(i,j)));
                correlationn=correlationn+((1./glcm_std).*((i-glcm_mean).*(j-glcm_mean).*(glcm(i,j))));
                dissimilarity=dissimilarity+(abs(i-j).*glcm(i,j));
                energyy=energyy+((glcm(i,j)).^2);
                entropyy=entropyy+(((-1.*glcm(i,j)).*log2(glcm(i,j))));
                harralick_corr=harralick_corr+(1./(Std_x).*((i.*j.*glcm(i,j))-Mx));
                
        end
    end
end

fprintf('\n autocorrelation is %f \n',autocorr);
fprintf('\n cluster prominance is %f \n',cluster_prominance);
fprintf('\n cluster tendancy is %f \n',cluster_tendancy);
fprintf('\n cluster shade is %f \n',cluster_shade);
fprintf('\n contrast is %f \n',contrast);
fprintf('\n correlation is %f \n',correlationn);
fprintf('\n dissimilarity is %f \n',dissimilarity);
fprintf('\n energy is %f \n',energyy);
fprintf('\n entropyy is %f \n',entropyy);
fprintf('\n harralick_corr is %f \n',harralick_corr);
