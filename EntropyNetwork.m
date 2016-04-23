%% Create and teach Izhikevich Neural Networks Visual Stimuli
seednums = randi(99999,[1,25]);

%create mexian hat filter for receptive fields
[x,y] = meshgrid(-12:12,-12:12);
sig1 = 2;
sig2 = sig1*1.6;
h1 = 1/(sig1*sqrt(2*pi))*exp(-(x.^2+y.^2)/2/sig1^2);
h2 = 1/(sig2*sqrt(2*pi))*exp(-(x.^2+y.^2)/2/sig2^2);
h1 = h1/sum(sum(h1));
h2 = h2/sum(sum(h2));
h = h1-h2;
clear h1 h2
h = h/max(max(h));

%locations for receptive fields for LGN excitatory (exc) and inhibitory (inh) cells
[x,y] = meshgrid(22:22/2:256);
excind = sub2ind([256 256],y,x);
excind = excind(1:end);
[x,y] = meshgrid(25:25:256);
inhind = sub2ind([256 256],y,x);
inhind = inhind(1:end);
clear x y

for sdn = 1:length(seednums);
    tic
    rand('seed',seednums(sdn));
    
    img_dir = 'C:\Users\seth.koenig\Documents\MATLAB\VPLT\ImageFiles\';
    sets = [126:135 142:144];
    imagesperset = 200;
    numimages = 10000;
    
    % spnet.m: Spiking network with axonal conduction delays and STDP
    % Created by Eugene M.Izhikevich.                February 3, 2004
    % Modified to allow arbitrary delay distributions.  April 16,2008
    D=5;                  % maximal conduction delay
    % excitatory neurons   % inhibitory neurons      % total number
    Ne=1068;                Ni=100;                   N=Ne+Ni;
    a=[0.02*ones(Ne,1);    0.1*ones(Ni,1)];
    d=[   8*ones(Ne,1);    2*ones(Ni,1)];
    sm=10;                 % maximal synaptic strength
    M=10;%round(0.1*N);                 % number of synapses per neuron
    
    % post=ceil([N*rand(Ne,M);Ne*rand(Ni,M)]);
    % Take special care not to have multiple connections between neurons
    delays = cell(N,D);
    post = NaN(N,M);
    for i=1:N
        if i <= 584 %feedfroorward "LGN"-like cells
            p=randperm(584)+584;
            post(i,:)=p(1:M);
            for j=1:M
                delays{i, ceil(D*rand)}(end+1) = j;  % Assign random exc delays
            end
            
%             post(i,1)=i+584;
%             delays{i,5}(end+1) = 1;  % Assign static delay from LGN to V1
        elseif i<= Ne %V1 excitatory cells
            p=randperm(584)+584;
            post(i,:)=p(1:M);
            for j=1:M
                delays{i, ceil(D*rand)}(end+1) = j;  % Assign random exc delays
            end
        else % V1 inhibitory cells
            p=randperm(484)+584;
            post(i,:)=p(1:M);
            delays{i,1}=1:M;                    % all inh delays are 1 ms.
        end
    end
    for i=Ne+1:N
        
    end
    
    s=[6*ones(Ne,M);-5*ones(Ni,M)];         % synaptic weights
    s0 = s;
    sd=zeros(N,M);                          % their derivatives
    
    % Make links at postsynaptic targets to the presynaptic weights
    pre = cell(N,1);
    aux = cell(N,1);
    for i=1:Ne
        for j=1:D
            for k=1:length(delays{i,j})
                pre{post(i, delays{i, j}(k))}(end+1) = N*(delays{i, j}(k)-1)+i;
                aux{post(i, delays{i, j}(k))}(end+1) = N*(D-1-j)+i; % takes into account delay
            end
        end
    end
    
    STDP = zeros(N,101+D);
    v = -65*ones(N,1);                      % initial values
    u = 0.2.*v;                             % initial values
    firings=[-D 0];                         % spike timings
    
    for image = 1:numimages
        
        img = imread([img_dir 'SET' num2str(sets(randi(length(sets),1))) '\' num2str(randi(200,1)) '.bmp']);
        img = double(rgb2gray(img));
        img = imfilter(img,h);
        img = img - mean2(img);
        img = img/max(max(img));
        img(img < 0) = 0;
        img = img*20;
        img(isnan(img)) = 0; %for some reason some images had nans
        
        I0 = zeros(N,1);
        I0(1:484,1)=img(excind);
        I0(485:584)=img(inhind);
        
        for t=1:20                         % simulation of 50 ms
            
            I = I0 + 5*rand(length(I0),1); %img input plus noise
            I(I > 20) = 20;
            
            fired = find(v>=30);                % indices of fired neurons
            v(fired)=-65;
            u(fired)=u(fired)+d(fired);
            STDP(fired,t+D)=0.1;
            for k=1:length(fired)
                sd(pre{fired(k)})=sd(pre{fired(k)})+STDP(N*t+aux{fired(k)});
            end
            firings=[firings;t*ones(length(fired),1),fired];
            k=size(firings,1);
            while firings(k,1)>t-D
                del=delays{firings(k,2),t-firings(k,1)+1};
                ind = post(firings(k,2),del);
                I(ind)=I(ind)+s(firings(k,2), del)';
                sd(firings(k,2),del)=sd(firings(k,2),del)-1.2*STDP(ind,t+D)';
                k=k-1;
            end
            v=v+0.5*((0.04*v+5).*v+140-u+I);    % for numerical
            v=v+0.5*((0.04*v+5).*v+140-u+I);    % stability time
            u=u+a.*(0.2*v-u);                   % step is 0.5 ms
            STDP(:,t+D+1)=0.95*STDP(:,t+D);     % tau = 20 ms
        end
%         plot(firings(:,1),firings(:,2),'.');
%         axis([0 50 0 N]); drawnow;
        STDP(:,1:D+1)=STDP(:,101:101+D);
        ind = find(firings(:,1) > 101-D);
        firings=[-D 0;firings(ind,1)-100,firings(ind,2)];
        s(1:Ne,:)=max(0,min(sm,0.01+s(1:Ne,:)+sd(1:Ne,:)));
        sd=0.9*sd;
    end
    toc
    save(['Network_seed_' num2str(seednums(sdn))],'s','s0','post','delays','pre','aux')
end
emailme('Networks Done running')