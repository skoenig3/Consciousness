% Take learned artificial neural networks made by EntropyNetwork.m and
% get their response to the visual stimlui
n = 9; %9 so square indexes
M = (dec2bin(0:(2^n)-1)=='1') ; % all possible combinations of firing pattersn for n neurons

img_dir = 'C:\Users\seth.koenig\Documents\MATLAB\VPLT\ImageFiles\';
sets = [126:135 142:144];
imagesperset = 200;

[x,y] = meshgrid(-12:12,-12:12);
sig1 = 2;
sig2 = sig1*1.6;
h1 = 1/(sig1*sqrt(2*pi))*exp(-(x.^2+y.^2)/2/sig1^2);
h2 = 1/(sig2*sqrt(2*pi))*exp(-(x.^2+y.^2)/2/sig2^2);
h1 = h1/sum(sum(h1));
h2 = h2/sum(sum(h2));
h = h1-h2;
clear h1 h2 sig1 sig2
h = h/max(max(h));

D=5;                  % maximal conduction delay
% excitatory neurons   % inhibitory neurons      % total number
Ne=1068;                Ni=100;                   N=Ne+Ni;
a=[0.02*ones(Ne,1);    0.1*ones(Ni,1)];
d=[   8*ones(Ne,1);    2*ones(Ni,1)];

%locations for receptive fields for excitatory (exc) and inhibitory (inh) cells
[x,y] = meshgrid(22:22/2:256);
excind = sub2ind([256 256],y,x);
excind = excind(1:end);
[x,y] = meshgrid(25:25:256);
inhind = sub2ind([256 256],y,x);
inhind = inhind(1:end);
clear x y

groups = [];
%group 1 centerish points
[xx,yy] = meshgrid(11:13);
ind = sub2ind([22,22],xx,yy);
groups(1,:) = ind(1:end);
%group 2 northern points
[xx,yy] = meshgrid(11:13,1:3);
ind = sub2ind([22,22],xx,yy);
groups(2,:) = ind(1:end);
%group 3 northeast points
[xx,yy] = meshgrid(20:22,1:3);
ind = sub2ind([22,22],xx,yy);
groups(3,:) = ind(1:end);
%group 4 eastern points
[xx,yy] = meshgrid(20:22,11:13);
ind = sub2ind([22,22],xx,yy);
groups(4,:) = ind(1:end);
%group 5 south eastern points
[xx,yy] = meshgrid(20:22,20:22);
ind = sub2ind([22,22],xx,yy);
groups(5,:) = ind(1:end);
%group 6 southern points
[xx,yy] = meshgrid(11:13,20:22);
ind = sub2ind([22,22],xx,yy);
groups(6,:) = ind(1:end);
%group 7 south western points
[xx,yy] = meshgrid(1:3,20:22);
ind = sub2ind([22,22],xx,yy);
groups(7,:) = ind(1:end);
%group 8 western points
[xx,yy] = meshgrid(1:3,11:13);
ind = sub2ind([22,22],xx,yy);
groups(8,:) = ind(1:end);
%group 9 north western points
[xx,yy] = meshgrid(1:3,1:3);
ind = sub2ind([22,22],xx,yy);
groups(9,:) = ind(1:end);
ind = 1:484;
ind(groups(1:end)) = []; %remove used indices
for g = 10:18 %take random ind
    r = randperm(length(ind));
    groups(g,:) = ind(r(1:n));
    ind(r(1:n)) = [];
end
clear xx yy ind g r

w = what;
w = w.mat;
for network = 1:size(w,1);
    learned_network_response = zeros(size(groups,1),size(M,1));
    unlearned_network_response = zeros(size(groups,1),size(M,1));
    random_network_response = zeros(size(groups,1),size(M,1));
    if ~isempty(strfind(w{network},'Network'));
        load(w{network})
        
        randind = randperm(484*10);
        sr = s(585:end,:); %shuffle weights in s
        sr = sr(randind);
        sr = reshape(sr,[484,10]);
        sr = [s(1:584,:); sr; -5*ones(100,10)];% randomly weighted network
        
        for set = 1:length(sets)
            for image = 1:imagesperset
                
                img = imread([img_dir 'SET' num2str(sets(set)) '\' num2str(image) '.bmp']);
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
                
                %---Simulate the Learned Network---%
                v = -65*ones(N,1);                      % initial values
                u = 0.2.*v;                             % initial values
                firings=[-D 0];                         % spike timings
                for t=1:50                         % simulation of 50 ms
                    I = I0 + 0*rand(length(I0),1); %img input plus noise
                    fired = find(v>=30);                % indices of fired neurons
                    v(fired)=-65;
                    u(fired)=u(fired)+d(fired);
                    firings=[firings;t*ones(length(fired),1),fired];
                    k=size(firings,1);
                    while firings(k,1)>t-D
                        del=delays{firings(k,2),t-firings(k,1)+1};
                        ind = post(firings(k,2),del);
                        I(ind)=I(ind)+s(firings(k,2), del)';
                        k=k-1;
                    end
                    v=v+0.5*((0.04*v+5).*v+140-u+I);    % for numerical
                    v=v+0.5*((0.04*v+5).*v+140-u+I);    % stability time
                    u=u+a.*(0.2*v-u);                   % step is 0.5 ms
                end
                
                firings(1,:) = [];
                firings = unique(firings);
                for group = 1:size(groups,1)
                    total = zeros(1,size(groups,2));
                    for neuron = 1:size(groups,2)
                        if ~isempty(find(firings == groups(group,neuron)));
                            total(neuron) = 1;
                        end
                    end
                    [~,indx]=ismember(total,M,'rows');
                    learned_network_response(group,indx) = learned_network_response(group,indx) + 1;
                end
                
                %---Simulate the Unlearned Network---%
                v = -65*ones(N,1);                      % initial values
                u = 0.2.*v;                             % initial values
                firings=[-D 0];                         % spike timings
                for t=1:50                         % simulation of 50 ms
                    I = I0 + 0*rand(length(I0),1); %img input plus noise
                    fired = find(v>=30);                % indices of fired neurons
                    v(fired)=-65;
                    u(fired)=u(fired)+d(fired);
                    firings=[firings;t*ones(length(fired),1),fired];
                    k=size(firings,1);
                    while firings(k,1)>t-D
                        del=delays{firings(k,2),t-firings(k,1)+1};
                        ind = post(firings(k,2),del);
                        I(ind)=I(ind)+s0(firings(k,2), del)';
                        k=k-1;
                    end
                    v=v+0.5*((0.04*v+5).*v+140-u+I);    % for numerical
                    v=v+0.5*((0.04*v+5).*v+140-u+I);    % stability time
                    u=u+a.*(0.2*v-u);                   % step is 0.5 ms
                end
                
                firings(1,:) = [];
                firings = unique(firings);
                for group = 1:size(groups,1)
                    total = zeros(1,size(groups,2));
                    for neuron = 1:size(groups,2)
                        if ~isempty(find(firings == groups(group,neuron)));
                            total(neuron) = 1;
                        end
                    end
                    [~,indx]=ismember(total,M,'rows');
                    unlearned_network_response(group,indx) = unlearned_network_response(group,indx) + 1;
                end
                
                %---Simulate the Random Network---%
                v = -65*ones(N,1);                      % initial values
                u = 0.2.*v;                             % initial values
                firings=[-D 0];                         % spike timings
                for t=1:50                         % simulation of 50 ms
                    I = I0 + 0*rand(length(I0),1); %img input plus noise
                    fired = find(v>=30);                % indices of fired neurons
                    v(fired)=-65;
                    u(fired)=u(fired)+d(fired);
                    firings=[firings;t*ones(length(fired),1),fired];
                    k=size(firings,1);
                    while firings(k,1)>t-D
                        del=delays{firings(k,2),t-firings(k,1)+1};
                        ind = post(firings(k,2),del);
                        I(ind)=I(ind)+sr(firings(k,2), del)';
                        k=k-1;
                    end
                    v=v+0.5*((0.04*v+5).*v+140-u+I);    % for numerical
                    v=v+0.5*((0.04*v+5).*v+140-u+I);    % stability time
                    u=u+a.*(0.2*v-u);                   % step is 0.5 ms
                end
                
                firings(1,:) = [];
                firings = unique(firings);
                for group = 1:size(groups,1)
                    total = zeros(1,size(groups,2));
                    for neuron = 1:size(groups,2)
                        if ~isempty(find(firings == groups(group,neuron)));
                            total(neuron) = 1;
                        end
                    end
                    [~,indx]=ismember(total,M,'rows');
                    random_network_response(group,indx) = random_network_response(group,indx) + 1;
                end
                
            end
        end
        save(['Analysis_' w{network}(14:end-4)],'learned_network_response',...
            'unlearned_network_response','random_network_response')
    end
end