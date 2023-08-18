% Sample data
load('ex3data1.mat');


% Convert species labels to numerical classes
unique_classes = unique(y);
num_classes = numel(unique_classes);




% Number of features and classes
num_features = size(X, 2);

% Initialize theta matrix to store parameters for each classifier
theta_matrix = zeros(num_features + 1, num_classes);

% Add a bias term to the features (intercept)
X = [ones(size(X, 1), 1), X];

% Train One-vs-All classifiers
for class_idx = 1:num_classes
    class_label = unique_classes(class_idx);
    y_class = (y == class_idx);

    % Define the cost function and gradient for logistic regression
    costFunction = @(theta) lrCostFunction(theta, X, y_class, 0);
    
    % Initial guess for theta
    initial_theta = zeros(num_features + 1, 1);
    
    % Optimize theta using fmincg
    options = optimset('GradObj', 'on', 'MaxIter', 50);
    [theta, ~] = fmincg(costFunction, initial_theta, options);

    % Store theta for the current class
    theta_matrix(:, class_idx) = theta;
end

% % Make predictions using the trained model
% probabilities = sigmoid(X * theta_matrix);
% [~, predicted_class] = max(probabilities, [], 2);
% predicted_class_labels = cellfun(@(x) keys(class_map, x), num2cell(predicted_class));
% 
% % Display the accuracy of the model
% accuracy = sum(strcmp(y, predicted_class_labels)) / numel(y);
% disp(['Model Accuracy: ', num2str(accuracy)]);
