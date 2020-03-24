clear all
filename = 'input_data/3mic.wav';
outfile = 'output_data/3mic_out.wav';
% read file
[x, sample_rate]= audioread( filename);
% =============== Initialize variables ===============
len=512; % Frame size in samples
PERC=50; % window overlap in percent of frame size
len1=floor(len*PERC/100);
len2=len-len1;
win=hamming(len);  % define window
nFFT=len;
%--- allocate memory and initialize various variables
img=sqrt(-1);
x_old=zeros(len1,1);
Nframes=floor(length(x)/len2)-floor(len/len2);
xfinal=zeros(Nframes*len2,1);
%=================  Start Processing =================
k=1;
for n=1:Nframes
    insign=win.*x(k:k+len-1);
    
    % Take fourier transform of frame
    spec=fft(insign,nFFT);
    sig=abs(spec); % compute the magnitude

    % Take inverse fourier transform of frame
    xi_w= ifft( sig .* exp(img*angle(spec)));
    xi_w= real( xi_w);

    % --------- Overlap and add ---------------
    xfinal(k:k+ len2-1)= x_old+ xi_w(1:len1);
    x_old= xi_w(len1+ 1: len);

    k=k+len2;
end
%======================================================
audiowrite(outfile,xfinal,sample_rate);

