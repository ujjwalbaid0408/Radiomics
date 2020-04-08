clc;
clear all; close all;


% % Load original image.
% load nbarb1;
% 
% % Image coding.
% nbcol = size(map,1);
% cod_X = wcodemat(X,nbcol);
% 
% % Visualize the original image.
% subplot(221)
% image(cod_X)
% title('Original image');
% colormap(map)



% Perform SWT decomposition
% of X at level 3 using sym4.
X = imread('test.jpg');
X = imresize(X, [ 256,256])

[ca,chd,cvd,cdd] = swt2(X,1,'sym4');


figure, imshow(ca,[]);
figure, imshow(chd,[]);
figure, imshow(cvd,[]);
figure, imshow(cdd,[]);

% Visualize the decomposition.
% 
% for k = 1:1
%     % Images coding for level k.
%     cod_ca  = wcodemat(ca(:,:,k),nbcol);
%     cod_chd = wcodemat(chd(:,:,k),nbcol);
%     cod_cvd = wcodemat(cvd(:,:,k),nbcol);
%     cod_cdd = wcodemat(cdd(:,:,k),nbcol);
%     decl = [cod_ca,cod_chd;cod_cvd,cod_cdd];
% 
%     % Visualize the coefficients of the decomposition
%     % at level k.
%     subplot(2,2,k+1)
%     image(decl)
% 
%     title(['SWT dec.: approx. ', ...
%    'and det. coefs (lev. ',num2str(k),')']);
%     colormap(map)
% end
% % Editing some graphical properties,
% % the following figure is generated.
