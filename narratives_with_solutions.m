%% narratives.m
% real-world human data 2020 spring, narratives tutorial
% hongmi lee 4/2/2020

% In this tutorial, you will generate narrative networks based on the semantic
% similarity between events in the narratives. The data provided (in the 'narrativesdata'
% folder) is the text descriptions of the three short movies you watched in the 
% class on 3/10 and the sentence embeddings of those text descriptions
% generated using Google's Universal Sentence Encoder (USE). Each row of
% the text files and the csv files represent each movie event (segmented by a human annotator).
% Each USE embedding has 512 features (numbers). By looking at the
% similarity between the USE embeddings, we can examine the inter-event
% structure of the movie plots. Specifically, we are interested in the
% centrality of events, which might reflect the relative importance of
% individual events in the plot. High centrality means the event is somehow
% semantically related to many other events in the story.  

%% 1. cd to the narratives directory 

clc;
clearvars;
close all;

% change this for your computer
basepath ='/Users/hlee239/Desktop/lecture/RWHDweek7/narratives_tutorial'; 
datapath = fullfile(basepath,'narrativesdata');
addpath(basepath);
cd(basepath);

%% 2. load USE embeddings and create similarity matrices
% This part loads in the USE embeddings in csv files (row=movie events, column=512
% features) for each movie and compute the cosine similarity between the embeddings (events). 
% In the event X event similarity matrices, high similarity event pairs
% are shown in yellow and low similarity pairs are shown in blue (with the parula colormap). 
% The similarity matrices show that different movies have different similarity structures.

% titles of the movies to be loaded
movietitles = {'the_record','the_shoe','the_black_hole'};

% similarity matrices will be saved in a cell
similmat = cell(1,numel(movietitles));

% create & show similarity matrices 
figure(1); 
for m = 1:numel(movietitles)
    
    % load USE embeddings for each movie
    embeddings = load(fullfile(datapath,['embeddings_' movietitles{m}  '.csv']));
    
    % cosine similarity between embeddings
    similmat{m} = 1-pdist2(embeddings,embeddings,'cosine');
    
    % create a subplot
    subplot(1,numel(movietitles),m);
    
    % show the similarity matrix and the colorbar
    imagesc(similmat{m}); caxis([0 1]); colorbar;
    
    % change axis properties
    set(gca,'XTick',[],'YTick',[],'FontSize',14);
    xticklabels(''); yticklabels(''); 
    
    % show the movie title (after replacing '_' with blank space) 
    title(strrep(movietitles{m},'_', ' '));
    
end

%% 3. generate narrative networks & compute event centrality 
% Using the similarity matrices created above, this part generates the
% narrative network for each movie. Each node of the narrative network is a
% movie event, and the edge between two nodes (events) is the cosine similarity
% between the USE embeddings of those events. You can threshold the edge
% weights so that only relatively strong connections are included in the narrative network.  
% The centrality of each movie event is measured as the pagerank of the node.

% edge weight threshold (no connection between nodes if the cosine similarity is lower than this)
thr = 0.3; % change this as you want 

% graphs, node names, centrality metrics to be saved
G = cell(1,numel(movietitles));
pagerank = cell(1,numel(movietitles));
nodenames = cell(1,numel(movietitles));

% create narrative networks (graphs)
for m = 1:numel(movietitles)
    
    % number of events (USE embedding vectors) within the movie
    numevent = size(similmat{m},1);
    
    % node names (event number as a string) saved in a cell
    nodenames{m} = cell(1,numevent);
    for n = 1:numevent
        nodenames{m}{n} = num2str(n);
    end
    
    % indices for the lower triangle part of the similarity matrix
    ltriidx = find(~triu(ones(numevent,numevent)));
    
    % all possible node pairs (first node - second node)
    [idx1,idx2] = meshgrid(1:numevent,1:numevent);
    idx1 = idx1(ltriidx); % first node numbers
    idx2 = idx2(ltriidx); % second node numbers
    
    % edge weights (cosine similarity between USE embeddings)
    weights = similmat{m}(ltriidx);
    
    % threshold edge weights
    cutidx = find(weights <= thr); % indices of node pairs to be excluded
    idx1(cutidx) = [];
    idx2(cutidx) = [];
    weights(cutidx) = [];
    
    % create graph
    G{m} = graph(idx1,idx2,weights,nodenames{m});
    
    % pagerank centrality of each event
    pagerank{m} = centrality(G{m},'pagerank','Importance',G{m}.Edges.Weight);
    
end

%% 4. plot narrative networks 
% This part visualizes the narrative networks created above. The network structure
% reflects the similarity structure observed in part 2. Higher-centrality
% event nodes (circles) are larger than lower-centrality nodes. The color
% of edges (lines) reflects the strength of connections between nodes (darker = stronger).

% network plotting type
layout = 'force'; % change this as you want ('circle''layered''subspace')

% show narrative networks (graphs)
figure(2);
for m = 1:numel(movietitles)
    
    % create a subplot
    subplot(1,numel(movietitles),m);
    
    % node size (weighted by node centrality)
    nodesize = pagerank{m}*150;
    
    % draw networks
    colormap(flipud(colormap('gray'))); % for edge colors (stronger connection = darker) 
    plot(G{m},'LineWidth',2,'Layout',layout,'NodeLabel',nodenames{m},...
        'Marker','o','MarkerSize',nodesize,'EdgeCData',G{m}.Edges.Weight);
    title(strrep(movietitles{m},'_', ' '));
    
    % change axis properties
    set(gca,'box','off','FontSize',14);
    ax1 = gca; ax1.YAxis.Visible = 'off'; ax1.XAxis.Visible = 'off';
    
end

%% 5. plot event centrality and mark high vs. low centrality events
% This part plots the time course of the event-by-event centrality. 
% The highest-centrality events are marked with *, and the lowest-centrality
% events are marked with o. You can see that the event centrality is not
% consistent across all events in a story.

% mark top N high/low centrality events
nmark = 2; % change this as you want

figure(3);
for m = 1:numel(movietitles)    
    
    % sort the centrality (high to low)
    [~,highidx] = sort(pagerank{m},'descend');
    
    % create a subplot
    subplot(1,numel(movietitles),m);
    
    % plot pagerank centrality
    plot(pagerank{m},'LineWidth',2); hold on;
    
    % mark top N high centrality events
    plot(highidx(1:nmark),pagerank{m}(highidx(1:nmark)),'k*');
    
    % mark top N low centrality events
    plot(highidx((end-nmark+1):end),pagerank{m}(highidx((end-nmark+1):end)),'ko');
    
    % axis labels
    xlabel('movie events');
    ylabel('centrality (pagerank)');
    title(strrep(movietitles{m},'_', ' '));

end

%% 6. check the text descriptions of the top N high/low centrality events
% Running this section will show the text descriptions of the high or low centrality events.
% It will also show the event number and the pagerank centrality of the
% events. Do you think the high centrality events are actually
% important/critical events of the movie plots?

% show top N high/low centrality events
nmark = 2; % change this as you want

for m = 1:numel(movietitles) 
    
    % sort the centrality (high to low)
    [~,highidx] = sort(pagerank{m},'descend');
    
    % load annotation text 
    textfile = extractFileText(fullfile(datapath,['annotation_' movietitles{m} '.txt']));
    
    % split the text (line by line)
    textlines = strsplit(textfile,'\n');
    
    % print out the events with the highest centrality 
    fprintf('\n\n%s: high centrality events\n\n', movietitles{m});
    for e = 1:nmark
        fprintf('event number: %2d  centrality: %1.4f  text: %s\n\n',...
            highidx(e), pagerank{m}(highidx(e)), textlines(highidx(e)));
    end
    
    % print out the events with the lowest centrality 
    fprintf('\n\n%s: low centrality events\n\n', movietitles{m});
    for e = 1:nmark
        fprintf('event number: %2d  centrality: %1.4f  text: %s\n\n',...
            highidx(end-e+1), pagerank{m}(highidx(end-e+1)), textlines(highidx(end-e+1)));
    end
end

%% exercise 1 (optional): create USE vectors yourself

% This is optional, but you can create USE vectors yourself using the 
% python script included in the tutorial folder (generate_USE_embedding.py)
% if you are familiar with python. The script loads in all text files in a
% folder and create USE vectors for each line of each text file.
% Then it saves the USE vectors in csv files (one csv file for one text file).  
% This script was used to generate the USE vectors for this tutorial, 
% using an older version of USE (version 3). You will first need to install 
% tensorflow (https://www.tensorflow.org/) and other  packages used in the script 
% to run the script, if you don't alreay have them. More information on USE can
% be found in https://tfhub.dev/google/universal-sentence-encoder/4. 

%% exercise 2: use word vectors 

%% 1. Use word vectors (word2vec, glove, etc.) to generate new 
% event-by-event sentence embeddings of the three movie annotations. 
% First preprocess the text to extract content words that exist in the word embedding model 
% that you choose to use (that is, exclude the words whose pre-trained embedding
% vectors do not exist). Then generate word vectors for each of the content words, 
% and average the word vectors across the content words within each movie event description. 

% clc;
% clearvars;
% close all;

% basepath ='/Users/hlee239/Desktop/lecture/RWHDweek7/narratives_tutorial'; 
% datapath = fullfile(basepath,'narrativesdata');
% addpath(basepath);
% cd(basepath);
% 
% movietitles = {'the_record','the_shoe','the_black_hole'};

%% choose a word embedding:

emb_choice = 'glove';

switch emb_choice
    
    case 'word2vec'
        % pre-trained word2vec word embedding (from Google, trained on Google News)
        % description & code: https://code.google.com/archive/p/word2vec/
        % direct download of FULL 3million word embeddings: https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit
        % SLIM 300k word embeddings: https://github.com/eyaler/word2vec-slim
        
        filename = "GoogleNews-vectors-negative300-SLIM.mat";
        load(filename);
        
    case 'glove'
        % pre-trained GloVe word embedding (from Stanford)
        % https://nlp.stanford.edu/projects/glove/
        % filename = "glove.6B.300d";
        filename = "glove.6B.50d"; % use this smaller one if the 300d is slow on your computer
        if exist(filename + '.mat', 'file') ~= 2
            emb = readWordEmbedding(filename + '.txt');
            save(filename + '.mat', 'emb', '-v7.3');
        else
            load(filename + '.mat')
        end
        
    case 'wikipedia'
        % MATLAB's pre-trained word embedding (unknown method, trained on Wikipedia)
        
        filename = "exampleWordEmbedding.vec";
        emb = readWordEmbedding(filename);
end

%% create word vector-based sentence embeddings

allembeddings = cell(1, numel(movietitles));

for m = 1:numel(movietitles)
    
    % load text and preprocess
    textfile = extractFileText(fullfile(datapath,['annotation_' movietitles{m} '.txt']));
    textdata_proc = split(textfile,newline);
    textdata_proc = erasePunctuation(textdata_proc);
    textdata_proc = lower(textdata_proc);
    textdata_proc = tokenizedDocument(textdata_proc);
    textdata_proc = removeWords(textdata_proc,stopWords);
    
    % for each line (movie event), extract content words & their embeddings
    allembeddings{m} = [];
    for l = 1:size(textdata_proc,1)
        
        bag = bagOfWords(textdata_proc(l));
        
        % sentence embedding = average of word embeddings
        allembeddings{m} = [allembeddings{m}; double(nanmean(word2vec(emb,bag.Vocabulary),1))];
    end

end

%% 2. Using the word vector-based sentence embeddings to re-create the following figures: 
% 1) similarity matrices 2) narrative networks 3) event-by-event centrality   
 
%%%%%%%%%%%%%%%%%%%%%%% similarity matrices
figure; 
for m = 1:numel(movietitles)
    
    % sentence embeddings created from word embeddings
    embeddings = allembeddings{m};
    
    % cosine similarity between embeddings
    similmat{m} = 1-pdist2(embeddings,embeddings,'cosine');
    
    % create a subplot
    subplot(1,numel(movietitles),m);
    
    % show the similarity matrix and the colorbar
    imagesc(similmat{m}); caxis([0.8 1]); colorbar; %%% note the caxis change (for visualization only)
    
    % change axis properties
    set(gca,'XTick',[],'YTick',[],'FontSize',14);
    xticklabels(''); yticklabels(''); 
    
    % show the movie title (after replacing '_' with blank space) 
    title(strrep(movietitles{m},'_', ' '));
end

%%%%%%%%%%%%%%%%%%%%% narrative networks

layout = 'force'; 
thr = 0.3; % result can vary depending on the edge weight threshold

% graphs, node names, centrality metrics to be saved
G = cell(1,numel(movietitles));
nodenames = cell(1,numel(movietitles));
pagerank_word = cell(1,numel(movietitles));

% create narrative networks (graphs)
figure;
colormap(flipud(colormap('gray'))); % for edge colors (stronger connection = darker) 
for m = 1:numel(movietitles)
    
    % number of events within the movie
    numevent = size(similmat{m},1);
    
    % node names (event number as a string) saved in a cell
    nodenames{m} = cell(1,numevent);
    for n = 1:numevent
        nodenames{m}{n} = num2str(n);
    end
    
    % indices for the lower triangle part of the similarity matrix
    ltriidx = find(~triu(ones(numevent,numevent)));
    
    % all possible node pairs (first node - second node)
    [idx1,idx2] = meshgrid(1:numevent,1:numevent);
    idx1 = idx1(ltriidx); % first node numbers
    idx2 = idx2(ltriidx); % second node numbers
    
    % edge weights (cosine similarity between USE embeddings)
    weights = similmat{m}(ltriidx);
    
    % threshold edge weights
    cutidx = find(weights <= thr); % indices of node pairs to be excluded
    idx1(cutidx) = [];
    idx2(cutidx) = [];
    weights(cutidx) = [];
    
    % create graph
    G{m} = graph(idx1,idx2,weights,nodenames{m});
    
    % pagerank centrality of each event
    pagerank_word{m} = centrality(G{m},'pagerank','Importance',G{m}.Edges.Weight);
    
    % plotting
    subplot(1,numel(movietitles),m);
    nodesize = pagerank_word{m}*150;
    plot(G{m},'LineWidth',2,'Layout',layout,'NodeLabel',nodenames{m},...
        'Marker','o','MarkerSize',nodesize,'EdgeCData',G{m}.Edges.Weight);
    title(strrep(movietitles{m},'_', ' '));
    set(gca,'box','off','FontSize',14);
    ax1 = gca; ax1.YAxis.Visible = 'off'; ax1.XAxis.Visible = 'off';
    
end

%%%%%%%%%%%%%%%%%%%%%% event-by-event centrality

% mark top N high/low centrality events
nmark = 2; % change this as you want

figure;
for m = 1:numel(movietitles)    
    
    % sort the centrality (high to low)
    [~,highidx] = sort(pagerank_word{m},'descend');
    
    % create a subplot
    subplot(1,numel(movietitles),m);
    
    % plot pagerank centrality
    plot(pagerank_word{m},'LineWidth',2); hold on;
    
    % mark top N high centrality events
    plot(highidx(1:nmark),pagerank_word{m}(highidx(1:nmark)),'k*');
    
    % mark top N low centrality events
    plot(highidx((end-nmark+1):end),pagerank_word{m}(highidx((end-nmark+1):end)),'ko');
    
    % axis labels
    xlabel('movie events');
    ylabel('centrality (pagerank)');
    title(strrep(movietitles{m},'_', ' '));

end

%% 3. Compare the structure of the word vector-based narrative networks and 
% the USE-based narrative networks. Do they look similar to each other?
% Compute the correlation between the word vector-based centrality and the 
% USE-based centrality. Are they positively correlated?   

% movie specific r
figure;
for m = 1:numel(movietitles)   

    [r, p] = corr(pagerank{m}, pagerank_word{m}) % pagerank{m} computed from the tutorial part above

    % (optional) scatterplot
    subplot(1,numel(movietitles),m);
    scatter(pagerank{m},pagerank_word{m});
    xlabel('USE-based centrality'); ylabel('glove-based centrality');
end
    
%% exercise 3: a story of your choice

% Collect a short story from the internet and create its narrative network.
% The story can be about a paragraph long, and each 'event' could be a
% sentence. If the story is longer than that, segment the story into
% several events based on its content. 

clc;
clearvars;
close all;

basepath ='/Users/hlee239/Desktop/lecture/RWHDweek7/narratives_tutorial'; 
datapath = fullfile(basepath,'narrativesdata');
addpath(basepath);
cd(basepath);

% choose a word embedding or use USE
filename = "exampleWordEmbedding.vec";
emb = readWordEmbedding(filename);

% load text and preprocess
storyfile = fullfile(datapath,'starwars.txt'); % each row is an event
textfile = extractFileText(storyfile);
textdata_proc = split(textfile,newline);
textdata_proc = erasePunctuation(textdata_proc);
textdata_proc = lower(textdata_proc);
textdata_proc = tokenizedDocument(textdata_proc);
textdata_proc = removeWords(textdata_proc,stopWords);

% for each line (movie event), extract content words & their embeddings
embeddings = [];
for l = 1:size(textdata_proc,1)
    
    bag = bagOfWords(textdata_proc(l));
    
    % sentence embedding = average of word embeddings
    embeddings = [embeddings; double(nanmean(word2vec(emb,bag.Vocabulary),1))];
end

% cosine similarity between embeddings
similmat = 1-pdist2(embeddings,embeddings,'cosine');

% show the similarity matrix and the colorbar
figure;
imagesc(similmat); caxis([0.5 1]); colorbar;
set(gca,'XTick',[],'YTick',[],'FontSize',14);
xticklabels(''); yticklabels('');

% create narrative network
thr = 0.3;
numevent = size(similmat,1);
nodenames = cell(1,numevent);
for n = 1:numevent
    nodenames{n} = num2str(n);
end

ltriidx = find(~triu(ones(numevent,numevent)));
[idx1,idx2] = meshgrid(1:numevent,1:numevent);
idx1 = idx1(ltriidx); % first node numbers
idx2 = idx2(ltriidx); % second node numbers

% edge weights
weights = similmat(ltriidx);

% threshold edge weights
cutidx = find(weights <= thr); % indices of node pairs to be excluded
idx1(cutidx) = [];
idx2(cutidx) = [];
weights(cutidx) = [];

% create graph
G = graph(idx1,idx2,weights,nodenames);

% pagerank centrality of each event
pagerank = centrality(G,'pagerank','Importance',G.Edges.Weight);

% plotting
figure;
layout = 'force';
colormap(flipud(colormap('gray'))); % for edge colors (stronger connection = darker)
nodesize = pagerank*150;
plot(G,'LineWidth',2,'Layout',layout,'NodeLabel',nodenames,...
    'Marker','o','MarkerSize',nodesize,'EdgeCData',G.Edges.Weight);
set(gca,'box','off','FontSize',14);
ax1 = gca; ax1.YAxis.Visible = 'off'; ax1.XAxis.Visible = 'off';

% mark top N high/low centrality events
nmark = 2; % change this as you want

% sort the centrality (high to low)
[~,highidx] = sort(pagerank,'descend');

% plot pagerank centrality
figure;
plot(pagerank,'LineWidth',2); hold on;
plot(highidx(1:nmark),pagerank(highidx(1:nmark)),'k*');
plot(highidx((end-nmark+1):end),pagerank(highidx((end-nmark+1):end)),'ko');
xlabel('events');
ylabel('centrality (pagerank)');

