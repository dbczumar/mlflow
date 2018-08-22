package org.mlflow.sagemaker;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;
import java.util.Optional;
import java.util.stream.Collectors;
import org.mlflow.mleap.MLeapLoader;
import org.mlflow.models.Model;

import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;
import org.eclipse.jetty.server.Server;
import org.eclipse.jetty.server.ServerConnector;
import org.eclipse.jetty.server.HttpConnectionFactory;
import org.eclipse.jetty.servlet.ServletHolder;
import org.eclipse.jetty.servlet.ServletContextHandler;
import org.eclipse.jetty.util.thread.QueuedThreadPool;

/** A RESTful webserver for {@link Predictor Predictors} that runs on the local host */
public class ScoringServer {
  public static final String RESPONSE_KEY_ERROR_MESSAGE = "Error";

  private static final int MINIMUM_SERVER_THREADS = 1;
  // Assuming an 8 core machine with hyperthreading
  private static final int MAXIMUM_SERVER_THREADS = 16;

  private enum RequestContentType {
    Csv("text/csv"),
    Json("application/json"),
    Invalid("invalid");

    private final String value;
    private static final Map<String, RequestContentType> BY_VALUE_MAP = new HashMap<>();

    static {
      for (RequestContentType inputType : RequestContentType.values()) {
        BY_VALUE_MAP.put(inputType.value, inputType);
      }
    }

    RequestContentType(String value) {
      this.value = value;
    }

    public static RequestContentType fromValue(String value) {
      if (BY_VALUE_MAP.containsKey(value)) {
        return BY_VALUE_MAP.get(value);
      } else {
        return RequestContentType.Invalid;
      }
    }
  }

  private final Server server;
  private final ServerConnector httpConnector;

  /**
   * Constructs a {@link ScoringServer} to serve the specified {@link Predictor} on the local host
   * on a randomly-selected available port
   */
  public ScoringServer(Predictor predictor) {
    this.server = new Server(new QueuedThreadPool(MAXIMUM_SERVER_THREADS, MINIMUM_SERVER_THREADS));
    this.server.setStopAtShutdown(true);

    this.httpConnector = new ServerConnector(this.server, new HttpConnectionFactory());
    this.server.addConnector(this.httpConnector);

    ServletContextHandler rootContextHandler = new ServletContextHandler(null, "/");
    rootContextHandler.addServlet(new ServletHolder(new ScoringServer.PingServlet()), "/ping");
    rootContextHandler.addServlet(
        new ServletHolder(new ScoringServer.InvocationsServlet(predictor)), "/invocations");
    this.server.setHandler(rootContextHandler);
  }

  /**
   * Loads the MLFlow model at the specified path as a {@link Predictor} and serves it on the local
   * host at the specified port
   *
   * @param modelPath The path to the MLFlow model to serve
   */
  public ScoringServer(String modelPath) throws PredictorLoadingException {
    this(loadPredictorFromPath(modelPath));
  }

  private static Predictor loadPredictorFromPath(String modelPath)
      throws PredictorLoadingException {
    Model config = null;
    try {
      config = Model.fromRootPath(modelPath);
    } catch (IOException e) {
      throw new PredictorLoadingException(
          "Failed to load the configuration for the MLFlow model at the specified path.");
    }
    return (new MLeapLoader()).load(config);
  }

  /**
   * Starts the scoring server locally on a randomly-selected, available port
   *
   * @throws IllegalStateException If the server is already active on another port
   * @throws ServerStateChangeException If the server failed to start and was inactive prior to the
   * invocation of this method
   */
  public void start() {
    // Setting port zero instructs Jetty to select a random port
    start(0);
  }

  /**
   * Starts the scoring server locally on the specified port
   *
   * @throws IllegalStateException If the server is already active on another port
   * @throws ServerStateChangeException If the server failed to start and was inactive prior to the
   * invocation of this method
   */
  public void start(int portNumber) {
    if (isActive()) {
      int activePort = this.httpConnector.getLocalPort();
      throw new IllegalStateException(String.format(
          "Attempted to start a server that is already active on port %d", activePort));
    }

    this.httpConnector.setPort(portNumber);
    try {
      this.server.start();
    } catch (Exception e) {
      throw new ServerStateChangeException(e);
    }
  }

  /**
   * Stops the scoring server
   *
   * @throws IllegalStateException If the server is already active on another port
   * @throws ServerStateChangeException If the server failed to start and was inactive prior to the
   * invocation of this method
   */
  public void stop() {
    try {
      this.server.stop();
      this.server.join();
    } catch (Exception e) {
      throw new ServerStateChangeException(e);
    }
  }

  /** @return `true` if the server is active (running), `false` otherwise */
  public boolean isActive() {
    return this.server.isStarted();
  }

  /**
   * @return Optional that either:
   * - Contains the port on which the server is running, if the server is active
   * - Is empty, if the server is not active
   */
  public Optional<Integer> getPort() {
    int boundPort = this.httpConnector.getLocalPort();
    if (boundPort >= 0) {
      return Optional.of(boundPort);
    } else {
      // The server connector port request returned an error code
      return Optional.empty();
    }
  }

  public static class ServerStateChangeException extends RuntimeException {
    ServerStateChangeException(Exception e) {
      super(e.getMessage());
    }
  }

  static class PingServlet extends HttpServlet {
    @Override
    public void doGet(HttpServletRequest request, HttpServletResponse response) {
      response.setStatus(HttpServletResponse.SC_OK);
    }
  }

  static class InvocationsServlet extends HttpServlet {
    private final Predictor predictor;

    InvocationsServlet(Predictor predictor) {
      this.predictor = predictor;
    }

    @Override
    public void doPost(HttpServletRequest request, HttpServletResponse response)
        throws IOException {
      RequestContentType contentType =
          RequestContentType.fromValue(request.getHeader("Content-type"));
      String requestBody =
          request.getReader().lines().collect(Collectors.joining(System.lineSeparator()));

      String responseContent = null;
      try {
        responseContent = evaluateRequest(requestBody, contentType);
      } catch (PredictorEvaluationException e) {
        response.setStatus(HttpServletResponse.SC_INTERNAL_SERVER_ERROR);
        responseContent = getErrorResponseJson(e.getMessage());
      } catch (InvalidRequestTypeException e) {
        response.setStatus(HttpServletResponse.SC_BAD_REQUEST);
      } catch (Exception e) {
        e.printStackTrace();
        response.setStatus(HttpServletResponse.SC_INTERNAL_SERVER_ERROR);
        responseContent =
            getErrorResponseJson("An unknown error occurred while evaluating the model!");
      } finally {
        response.getWriter().print(responseContent);
        response.getWriter().close();
      }
    }

    private String evaluateRequest(String requestContent, RequestContentType contentType)
        throws PredictorEvaluationException, InvalidRequestTypeException {
      switch (contentType) {
        case Json: {
          DataFrame parsedInput = null;
          try {
            parsedInput = DataFrame.fromJson(requestContent);
          } catch (UnsupportedOperationException e) {
            throw new PredictorEvaluationException(
                "This model does not yet support evaluating JSON inputs.");
          }
          DataFrame result = predictor.predict(parsedInput);
          return result.toJson();
        }
        case Csv: {
          DataFrame parsedInput = null;
          try {
            parsedInput = DataFrame.fromCsv(requestContent);
          } catch (UnsupportedOperationException e) {
            throw new PredictorEvaluationException(
                "This model does not yet support evaluating CSV inputs.");
          }
          DataFrame result = predictor.predict(parsedInput);
          return result.toCsv();
        }
        default:
          throw new InvalidRequestTypeException(
              "Invocations content must be of content type `application/json` or `text/csv`");
      }
    }

    private String getErrorResponseJson(String errorMessage) {
      String response =
          String.format("{ \"%s\" : \"%s\" }", RESPONSE_KEY_ERROR_MESSAGE, errorMessage);
      return response;
    }

    static class InvalidRequestTypeException extends Exception {
      InvalidRequestTypeException(String message) {
        super(message);
      }
    }
  }

  /**
   * Entrypoint for locally serving MLFlow models with the MLeap flavor using the {@link
   * ScoringServer}
   *
   * <p>This entrypoint expects the following arguments: 1. The path to the MLFlow model to serve.
   * This model must have the MLeap flavor. 2. (Optional) the number of the port on which to serve
   * the MLFlow model.
   */
  public static void main(String[] args) throws IOException, PredictorLoadingException {
    String modelPath = args[0];
    Optional<Integer> portNum = Optional.empty();
    if (args.length > 1) {
      portNum = Optional.of(Integer.parseInt(args[2]));
    }
    ScoringServer server = new ScoringServer(modelPath);
    try {
      server.start(8080);
    } catch (Exception e) {
      e.printStackTrace();
    }
  }
}
