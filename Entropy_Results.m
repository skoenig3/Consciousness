% Combine and Calculate the results from Entropy_Analysis.m

learned_entropy = NaN(2,25);
unlearned_entropy = NaN(2,25);
random_entropy = NaN(2,25);

count = 1;

w = what;
w = w.mat;
for file = 1:size(w,1);
    if ~isempty(strfind(w{file},'Analysis'));
        load(w{file})
        
        numgroups = size(learned_network_response,1);
        
        learned_struct_groups = NaN(1,numgroups/2); %groups of neurons near each other
        learned_unstruct_groups = NaN(1,numgroups/2);%random groups of neurons
        unlearned_struct_groups = NaN(1,numgroups/2); %groups of neurons near each other
        unlearend_unstruct_groups = NaN(1,numgroups/2);%random groups of neurons
        random_struct_groups = NaN(1,numgroups/2); %groups of neurons near each other
        random_unstruct_groups = NaN(1,numgroups/2);%random groups of neurons
        
        
        for group = 1:numgroups
            %make into pdf
            lnr = learned_network_response(group,:)/sum(learned_network_response(group,:));
            unr = unlearned_network_response(group,:)/sum(unlearned_network_response(group,:));
            rnr = random_network_response(group,:)/sum(random_network_response(group,:));
            
            %Shanon's entropy
            templ = -nansum(lnr.*log2(lnr));
            tempu = -nansum(unr.*log2(unr));
            tempr = -nansum(rnr.*log2(rnr));
            
            if group <= numgroups/2
                learned_struct_groups(group) = templ;
                unlearned_struct_groups(group) = tempu;
                random_struct_groups(group) = tempr;
            else
                learned_unstruct_groups(group-numgroups/2) = templ;
                unlearend_unstruct_groups(group-numgroups/2) = tempu;
                random_unstruct_groups(group-numgroups/2) = tempr;
            end
        end
        
        learned_entropy(1,count) = mean(learned_struct_groups);
        learned_entropy(2,count) = mean(learned_unstruct_groups);
        unlearned_entropy(1,count) = mean(unlearned_struct_groups);
        unlearned_entropy(2,count) = mean(unlearend_unstruct_groups);
        random_entropy(1,count) = mean(random_struct_groups);
        random_entropy(2,count) = mean(random_unstruct_groups);
        count = count+1;
    end
end