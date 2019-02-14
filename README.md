# IoTGAN
> This project proposes an on-device generative adversarial network (GAN) where each Internet of things (IoT) device pre-trains and shares its GAN weight vector over an ad-hoc local network.

[![NPM Version][npm-image]][npm-url]
[![Build Status][travis-image]][travis-url]
[![Downloads Stats][npm-downloads]][npm-url]

![](header.png)
Since in the on-device machine learning, the training data used by the model will be limited in terms of amount of data and mode coverage. In the IoT network, wider data coverage can be obtained through the broadcast data to sensor nodes in the network, so that the trained mode coverage, training accuracy and energy efficiency can be optimized. Since IoT devices often use personal sensing data, direct share raw data will involve security risk. In this paper we propose IoTGAN framework that shares weight vectors provided from the neighboring sensor nodes in the network.


![](header.png)
Figure 2 shows proposed IoT GAN Model. The model contains sensor data, GAN model and a classifier in a single node. Sensor data X and label Y can be used for training classifier directly. In order to train sharable weight vectors in the IoT network, an embedded buffer of GAN model uses received weight vector (WRX) to generate fake image and shuffling with sensor data for the further GAN training. After training, the GAN model transfer weight vectors (WTX) wirelessly to external nodes in the network. The GAN generated augmented dataset (X’, Y’) has been used to train classifier for extended mode coverage of classification.
## Installation

OS X & Linux:

```sh
npm install my-crazy-module --save
```

Windows:

```sh
edit autoexec.bat
```

## Usage example

A few motivating and useful examples of how your product can be used. Spice this up with code blocks and potentially more screenshots.

_For more examples and usage, please refer to the [Wiki][wiki]._

## Development setup

Describe how to install all development dependencies and how to run an automated test-suite of some kind. Potentially do this for multiple platforms.

```sh
make install
npm test
```

## Release History

* 0.2.1
    * CHANGE: Update docs (module code remains unchanged)
* 0.2.0
    * CHANGE: Remove `setDefaultXYZ()`
    * ADD: Add `init()`
* 0.1.1
    * FIX: Crash when calling `baz()` (Thanks @GenerousContributorName!)
* 0.1.0
    * The first proper release
    * CHANGE: Rename `foo()` to `bar()`
* 0.0.1
    * Work in progress

## Meta

Your Name – [@YourTwitter](https://twitter.com/dbader_org) – YourEmail@example.com

Distributed under the XYZ license. See ``LICENSE`` for more information.

[https://github.com/yourname/github-link](https://github.com/dbader/)

## Contributing

1. Fork it (<https://github.com/yourname/yourproject/fork>)
2. Create your feature branch (`git checkout -b feature/fooBar`)
3. Commit your changes (`git commit -am 'Add some fooBar'`)
4. Push to the branch (`git push origin feature/fooBar`)
5. Create a new Pull Request

<!-- Markdown link & img dfn's -->
[npm-image]: https://img.shields.io/npm/v/datadog-metrics.svg?style=flat-square
[npm-url]: https://npmjs.org/package/datadog-metrics
[npm-downloads]: https://img.shields.io/npm/dm/datadog-metrics.svg?style=flat-square
[travis-image]: https://img.shields.io/travis/dbader/node-datadog-metrics/master.svg?style=flat-square
[travis-url]: https://travis-ci.org/dbader/node-datadog-metrics
[wiki]: https://github.com/yourname/yourproject/wiki
