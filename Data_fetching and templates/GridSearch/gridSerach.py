from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSerachCV

def buid_classifier(optimizer):
    #Network

classifier = KerasClassifier(build_fn = build_classifier)
parameter = {
        #take a list of parameters
}
grid_search = GridSearchCV(estimator= ,param_grid = ,scoring = ,cv = )

grid_search = grid_search.fit(x,y)
best_parameters = grid_search.best_params
best_accuracy = grid_search.best_score