function result = candidateAlgStructure(name,dir,prefix,postfix,ext)
% Function creates the struct needed to evaluate algorithm
% Input:
% name - name of algorithm
% dir - directory of algorithm output
% prefix - prefix of saliency files - can be empty. (e.g. saliency_);
% postfix - postif of saliency files - can be empty. (e.g. _saleincy);
% ext - extension of saliency files (e.g. jpg,png)

result = struct('name', name, 'dir', dir,'prefix',prefix,'postfix',...
    postfix,'ext',ext);
end