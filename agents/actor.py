from keras import layers,models,optimizers
from keras import backend as K

class Actor():
    """Actor (Policy) Model"""

    def __init__(self,state_size,action_size,action_low,action_high):
        """Initialize and build actor model
        
        Params:
        ======
            state_size(int): size of the observation space
            action_size(int): size of the action space
            action_low(array): min value of each action dimension
            action_high(array): max value of each action dimension
        """

        self.state_size = state_size
        self.action_size = action_size
        self.action_high = action_high
        self.action_low = action_low
        self.action_range = self.action_high-self.action_low

        # Build actor model
        self.build_model()

    def build_model(self):
        """Return an actor policy network which maps states to actions"""

        # Define input layers
        states = layers.Input(shape=(self.state_size,),name='states')

        # Define hidden layers

        net = layers.Dense(units=400,kernel_regularizer=layers.regularizers.l2(1e-6))(states)
        net = layers.BatchNormalization()(net)
        net = layers.Activation("relu")(net)
        net = layers.Dense(units=300,kernel_regularizer=layers.regularizers.l2(1e-6))(net)
        net = layers.BatchNormalization()(net)
        net = layers.Activation("relu")(net)

        # Define output layers
        raw_actions = layers.Dense(units=self.action_size,activation='sigmoid',
            name='raw_actions',kernel_initializer=layers.initializers.RandomUniform(minval=-0.003, maxval=0.003))(net)

        # Scale raw actions to action space range

        actions = layers.Lambda(lambda x: (x*self.action_range)+self.action_low,
            name='actions')(raw_actions)

        # Create Keras model

        self.model = models.Model(inputs=states,outputs=actions)

        # Define loss function using action value (Q value) gradients

        action_gradients = layers.Input(shape=(self.action_size,))
        loss = K.mean(-action_gradients*actions)

        # Define optimizers
        optimizer = optimizers.Adam(lr = 0.0001)
        updates_op = optimizer.get_updates(params=self.model.trainable_weights, loss=loss)

        
        self.train_fn = K.function(
            inputs=[self.model.input, action_gradients, K.learning_phase()],
            outputs=[],
            updates=updates_op)
