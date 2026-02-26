import type { Configuration as WebpackConfiguration } from 'webpack';
import type { Configuration as DevServerConfiguration } from 'webpack-dev-server';
import { rules } from './webpack.rules';
import { plugins } from './webpack.plugins';

rules.push({
  test: /\.css$/,
  use: [{ loader: 'style-loader' }, { loader: 'css-loader' }],
});

type RendererConfig = WebpackConfiguration & Partial<DevServerConfiguration>;

export const rendererConfig: RendererConfig = {
  entry: './src/renderer.tsx',
  module: {
    rules,
  },
  plugins,
  resolve: {
    extensions: ['.js', '.ts', '.jsx', '.tsx', '.css'],
  },
  // when running the dev server we need to inject a CSP header
  // that allows connections back to our local API. webpack-dev-server
  // doesn't know about our custom index.html meta tag, so we add
  // it here to avoid CSP errors in development.
  devServer: {
    headers: {
      'Content-Security-Policy': "default-src 'self' 'unsafe-inline' 'unsafe-eval' data:; connect-src 'self' http://localhost:8000;",
    },
  },
};
