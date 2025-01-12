# Backstory
A few years ago I worked at a Fin-tech banking company. Their primary appeal was the data insights and analytics they collected from loaners using the [PSD2](https://en.wikipedia.org/wiki/Payment_Services_Directive) protocol. One of my tasks was to expand the analytics with new formulas and methods. Due to liability reasons I can't go too much into detail of the specifics of it all, but in short: I ended up using Logistic Regression as part of calculating the expected default % of loans (see [Investopedia: What is LGD?](https://www.investopedia.com/terms/l/lossgivendefault.asp) for more information.)

Of course, one could simply install Python and set up an LR model that way. However, at that point in time Python wasn't part of our software stack (yet) and we we're running a Monolithic architecture, which meant integrating Python would come with needless additional overhead for a small team of four.

There were some existing dependencies that provided Logistic Regression, yet came with many more mathematical features that we're simply not necessary for our use case. Remembering the lesson from "A Philosophy of Software Design", I decided to stick with the built-in math library of NodeJS and built the implementation myself.

# The implementation
For this implementation I'm using the "score" the company gave to customers, which was basically a grade that ran from -10 to +10, as the first variable. This score was the culmination of about a hundred different formulas. 
The second variable was the assigned credit limit (read: how much someone can loan) of a customer.  And the third and last variable was whether this customer was accepted

## Data matrix
Let's start with the basics: building your matrix.
```javascript
const createMatrix = async () => {
  let matrix = [];

  // define data sources
  const dataService = require('./data-service');

  // These are self-explanatory; so I won’t go into details.
  const averageScore = await getAverageScore();
  const customers = await getCustomers();

  try {
    for (const customer of customers) {
      const creditLimit = customer.currentCreditLimit;

      // It's a byte representation, this is needed for the LR model, 
      // hence why it's not named 'isAccepted' 
      let accepted;

      // Ours was based on a string value, because a customer could have
      // more than just these two statuses.
      switch (customer.acceptanceStatus) {
        case 'evaluationAccepted':
          accepted = 1;
          break;
        case 'evaluationRejected':
          accepted = 0;
          break;
      }
      
      const scoreDeviation = scoreDeviationFromAverage(averageScore, customer.score);
      matrix.push([scoreDeviation, creditLimit, accepted]);
    }
  } 
  catch (error) {
    // Implement your desired logger / error handling here
    logger.error(error);
    matrix = [];
  }
  finally {
    return matrix;
  }
}
```

### Calculating a deviation
In the matrix function we calculated the deviation. And while calculating a deviation isn't that hard, it does have some caveats, so here is my implementation:
```javascript
const scoreDeviationFromAverage = (average, score) => {
    let deviation = 0;

    if (average > score) {
      deviation = average - score;
    }
    else if (average < score) {
      deviation = score - average
    }

    if (score < 0) {
      deviation = deviation *-1;
    }

    return deviation;
}
```

## Optimizing costs and processing time
Now that we have our data matrix, it's time for some optimization. After researching the topic for a while I decided to make use of [gradient descent](https://en.wikipedia.org/wiki/Gradient_descent) as the optimization algorithm. Feel free to use whatever algorithm best suits your use case. 

Let's start building the algorithm:
```javascript
const gradientDescent = async(independentVariables, dependentVariable, theta, alpha, iterations) => {
  const size = dependentVariable.length;

  for (let i = 0; i < iterations; i++) {
    let hypothesis = sigmoid(math.evaluate(`independentVariables * theta`, {
      independentVariables,
      theta,
    }));

    theta = await math.evaluate(`theta - alpha / size * ((hypothesis - dependentVariable)' * independentVariables)'`, {
      theta,
      alpha,
      size,
      independentVariables,
      dependentVariable,
      hypothesis,
    });
  }

  return theta;
}

const sigmoid = (input) => {
  let output = math.evaluate(`1 ./ (1 + e.^-input)`, {
    input,
  });

  return output;
}
```

I think the Wikipedia article I linked will do a much better job at explaining the intricate details of the workings of this algorithm than I ever could. So if you are curious, I would recommend reading that.

Now that we have the cost optimization algorithm in place, we can calculate the "cost" to see what difference it has made:
```javascript
const calculateCost = (theta, independentVariables, dependentVariable) => {
  const size = dependentVariable.length;
  
  let hypothesis = sigmoid(math.evaluate(`independentVariables * theta`, {
    independentVariables,
    theta,
  }));

  const cost = math.evaluate(`(1 / size) * (-dependentVariable' * log(hypothesis) - (1 - dependentVariable)' * log(1 - hypothesis))`, {
    hypothesis,
    dependentVariable,
    size,
  });

  const gradient = math.evaluate(`(1 / size) * (hypothesis - dependentVariable)' * independentVariables`, {
    hypothesis,
    dependentVariable,
    size,
    independentVariables,
  });

  return { cost, gradient };
}
```

In the main function this will be on by default and logged automatically. I would recommend adding feature flags to this functionality as to not waste unnecessary resources in production.

## Training the Logistic Regression Model
We've assembled all the required functions to start training our model. The training model will have various variables that can be adjusted to your needs. I would highly recommend to read up on the meanings of these variables. As this tutorial is focused on building an implementation, I will not be explaining them.

Let's start building the training function:
```javascript
const trainCustomerAcceptanceLogisticRegressionModel = async(customerId) => {
  const matrix = await createMatrix();

  try {
    // ~ The learning rate of the algorithm
    const alpha = 0.01;

    // ~ Number of times the algorithm's parameters are updated
    const iterations = 10000;

    // ~ The initial theta parameters
    // ~ (local minimum to reduce iterations needed to find the global minumum)
    const thetaParameters = [[25], [5], [-30]];

    let independentVariables = math.evaluate('matrix[:, 1:2]', { matrix });
    let dependentVariable    = math.evaluate('matrix[:, 3]', { matrix });

    // ~ Add the intercept that defines DV when independent variables = 0
    independentVariables = math.concat(math.ones([dependentVariable.length, 1]).valueOf(), independentVariables);

    // ~ Calculate the value of theta (theta is the effect on DV when the dependent variables increase/decrease)
    const theta = await gradientDescent(independentVariables, dependentVariable, thetaParameters, alpha, iterations);

    // ~ Track the trained theta cost of the algorithm (where lower is better)
    const { cost }  = calculateCost(theta, independentVariables, dependentVariable);
    logger.log(`Cost of this training session: ${cost}`);

    if (theta && theta.length > 0) {
      // ~ Theta is a double array because this is how a vector is structured for mathjs
      const coefficientConstant                 = theta[0][0];
      const coefficientDeviationAverageScore    = theta[1][0];
      const coefficientCustomerLoan             = theta[2][0];

      // Implement a place to store the values of the customer here
      await db.query(
        /*SQL*/
        `
        INSERT INTO logisic_regression (customerId, coefficientConstant, coefficientDeviationAverageScore, coefficientCustomerLoan)
        VALUES ('${customerId}''${coefficientConstant}', '${coefficientDeviationAverageScore}', '${coefficientCustomerLoan}');
        `
      );
    }
  } 
  catch (error) {
    logger.error(error);
  }
}
```

## Predicting the outcome
Now that we have the coefficients ready we can execute the prediction for a customer with a simple function:
```javascript
const predictOutcome = (theta, independentVariables) => {
  const prediction = sigmoid(math.evaluate(`independentVariables * theta`, {
    independentVariables,
    theta,
  }));

  return prediction;
}
```

The function uses the theta from the logistic regression training model and the individual independent variables of a customer to predict the outcome of that customer. In my case it tried to predict if a customer would be accepted. With the target being the customers that were accepted and also eventually defaulted on a loan. When tracking this data over time, the LR algorithm provides insight into the performance of the analytics and future predictions of the customers the company had.

Author: Tim van Oudheusden
Published: 2025-01-12