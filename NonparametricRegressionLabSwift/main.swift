//
//  main.swift
//  NonparametricRegressionLabSwift
//
//  Created by Артём Бурмистров on 9/21/19.
//  Copyright © 2019 Артём Бурмистров. All rights reserved.
//

import Foundation
import CreateML
import CoreML

enum DistanceType: CaseIterable {
    case euclidean, manhattan, chebyshev
}

enum KernelType: CaseIterable {
    case /*uniform, triangular,*/ epanechnikov, /*quartic, triweight,
    tricube,*/ gaussian, cosine, logistic, sigmoid
}

enum WindowType: CaseIterable {
    case fixed, variable
}

struct Vector {
    var x: [Double]
    var y: Int?
    
    var count: Int { return x.count }
    subscript (index: Int) -> Double { return x[index] }
    
    static func ==(lhs: Vector, rhs: Vector) -> Bool {
        return lhs.x == rhs.x
    }
    
    static func !=(lhs: Vector, rhs: Vector) -> Bool {
        return lhs.x != rhs.x
    }
}

typealias DistanceFunc = (Vector, Vector) -> Double

func getDistanceFunc(distanceType: DistanceType) -> DistanceFunc {
    switch distanceType {
    case .euclidean:
        return {
            var res = 0.0
            for i in 0..<$0.count {
                res += pow($0[i] - $1[i], 2)
            }
            return sqrt(res)
        }
    case .manhattan:
        return {
            var res = 0.0
            for i in 0..<$0.count {
                res += abs($0[i] - $1[i])
            }
            return res
        }
    case .chebyshev:
        return {
            var res = 0.0
            for i in 0..<$0.count {
                res = max(abs($0[i] - $1[i]), res)
            }
            return res
        }
    }
}

typealias KernelFunc = (Double) -> Double

func getKernelFunc(kernelType: KernelType) -> KernelFunc {
    switch kernelType {
//    case .uniform:
//        return { abs($0) < 1 ? 1.0 / 2.0 : 0 }
//    case .triangular:
//        return { abs($0) < 1 ? 1 - abs($0) : 0 }
    case .epanechnikov:
        return { abs($0) < 1 ? 3.0 / 4.0 * (1 - pow($0, 2)) : 0 }
//    case .quartic:
//        return { abs($0) < 1 ? 15.0 / 16.0 * pow(1 - pow($0, 2), 2) : 0 }
//    case .triweight:
//        return { abs($0) < 1 ? 35.0 / 32.0 * pow(1 - pow($0, 2), 3) : 0 }
//    case .tricube:
//        return { abs($0) < 1 ? 70.0 / 81.0 * pow(1 - pow(abs($0), 3), 3) : 0 }
    case .gaussian:
        return { pow(M_E, -pow($0, 2) / 2 / sqrt(.pi * 2)) }
    case .cosine:
        return { abs($0) < 1 ? .pi / 4.0 * cos(.pi * $0 / 2) : 0 }
    case .logistic:
        return { 1.0 / (pow(M_E, $0) + 2 + pow(M_E, -$0)) }
    case .sigmoid:
        return { 2.0 / (.pi * (pow(M_E, $0) + pow(M_E, -$0))) }
    }
}

func argmax(points: [Vector], point: Vector, yCount: Int, h: Int,
            kernelFunc: KernelFunc, windowType: WindowType, distanceFunc: DistanceFunc) -> Int {
    var weights = [Double](repeating: 0, count: yCount)
    
    if (windowType == .fixed) {
        if h != 0 {
            for currentPoint in points {
                let distance = distanceFunc(point, currentPoint)
                let kernel = kernelFunc(distance / Double(h))
                weights[currentPoint.y!] += kernel
            }
        } else {
            for currentPoint in points {
                let distance = distanceFunc(point, currentPoint)
                let kernel = kernelFunc(distance / Double(h))
                
                if distance < Double.ulpOfOne {
                    weights[currentPoint.y!] += kernel
                }
            }
            
            if weights.allSatisfy({ $0 < Double.ulpOfOne }) {
                for currentPoint in points {
                    let kernel = kernelFunc(0)
                    
                    weights[currentPoint.y!] += kernel
                }
            }
        }
    } else {
        let kDistance = distanceFunc(point, points[h]);
        
        if (kDistance > Double.ulpOfOne) {
            for currentPoint in points {
                let distance = distanceFunc(point, currentPoint)
                
                if (distance <= kDistance) {
                    let kernel = kernelFunc(distance / kDistance)
                    weights[currentPoint.y!] += kernel
                }
            }
            
            if weights.allSatisfy({ $0 < Double.ulpOfOne }) {
                for currentPoint in points {
                    let distance = distanceFunc(point, currentPoint)
                    
                    if abs(distance - kDistance) < Double.ulpOfOne {
                        weights[currentPoint.y!] += Double(currentPoint.y!)
                    }
                }
            }
        } else {
            for currentPoint in points {
                let distance = distanceFunc(point, currentPoint)
                let kernel = kernelFunc(0);
                
                if (distance < Double.ulpOfOne) {
                    weights[currentPoint.y!] += kernel
                }
            }
            
            if weights.allSatisfy({ $0 < Double.ulpOfOne }) {
                for currentPoint in points {
                    let kernel = kernelFunc(0);
                    
                    weights[currentPoint.y!] += kernel
                }
            }
        }
    }
    
    var max = 0.0
    var maxIndex = 0
    for (index, elem) in weights.enumerated() {
        if elem > max {
            max = elem
            maxIndex = index
        }
    }
    
    return maxIndex
}

func leaveOneOut(points: [Vector], iters: Int, yCount: Int, h: Int,
                 kernelFunc: KernelFunc, windowType: WindowType, distanceFunc: DistanceFunc) -> [[Double]] {
    var res = [[Double]](repeating: .init(repeating: 0.0, count: yCount), count: yCount)
    
    for i in 0..<iters {
        let loo = points.filter({ $0 != points[i] }).sorted(by: { distanceFunc(points[i], $0) < distanceFunc(points[i], $1) })
        let predictedY = argmax(points: loo, point: points[i], yCount: yCount, h: h, kernelFunc: kernelFunc, windowType: windowType, distanceFunc: distanceFunc)
        res[points[i].y!][predictedY] += 1
    }
    
    return res
}

//if let dir = FileManager.default.urls(for: .downloadsDirectory, in: .userDomainMask).first {
//    let fileURL = dir.appendingPathComponent("dataset_26_nursery").appendingPathExtension("csv")
//
//    let input = try! String(contentsOf: fileURL).components(separatedBy: "\n").filter({ $0 != "" }).map { $0.components(separatedBy: ",") }.dropFirst()
//    var variations = [[(key: String, count: Int)]](repeating: [], count: input.first!.count)
//    var data = [Vector]()
//    for i in input.indices {
//        for j in input[i].indices {
//            if let ind = variations[j].firstIndex(where: { $0.key == input[i][j] }) {
//                variations[j][ind].count += 1
//            } else {
//                variations[j].append((input[i][j], 1))
//            }
//        }
//        let ind = variations.last!.firstIndex(where: { $0.key == input[i].last! })!
//        var elem = Vector(x: [], y: ind)
//        for (index, variation) in variations.dropLast().enumerated() {
//            elem.x.append(Double(variation.firstIndex(where: { $0.key == input[i][index] })!))
//        }
//        data.append(elem)
//    }
//
//    let allCounts = input.count
//
//    let means = variations.map { (variation: [(key: String, count: Int)]) -> Double in
//        var sum = 0.0
//        for (index, elem) in variation.enumerated() {
//            sum += Double(elem.count) * Double(index)
//        }
//        return sum / Double(allCounts)
//    }
//
//    var stds = [Double]()
//    for i in data[0].x.indices {
//        var sum = 0.0
//        for j in data.indices {
//            sum += pow(Double(data[j][i]) - means[i], 2) / Double(allCounts)
//        }
//        stds.append(sqrt(sum))
//    }
//
//    for (i, row) in data.enumerated() {
//        for (j, elem) in row.x.enumerated() {
//            data[i].x[j] = (elem - means[j]) / stds[j]
//        }
//    }
//
//    let logFile = dir.appendingPathComponent("scores").appendingPathExtension("txt")
//    let matrixFile = dir.appendingPathComponent("cfs").appendingPathExtension("txt")
//
//    for window in WindowType.allCases {
//        for kernel in KernelType.allCases {
//            for distance in DistanceType.allCases {
//                for h in 1..<20 {
//                    let confusionMatrix = leaveOneOut(points: data, iters: 1000 , yCount: 5, h: h,
//                                                      kernelFunc: getKernelFunc(kernelType: kernel),
//                                                      windowType: window,
//                                                      distanceFunc: getDistanceFunc(distanceType: distance))
//                    let text = "\(window) \(kernel) \(distance) \(h) \(calcMacroF1(confusionMatrix: confusionMatrix, all: 1000))\n"
//                    if let fileHandle = FileHandle(forWritingAtPath: logFile.path) {
//                        fileHandle.seekToEndOfFile()
//                        fileHandle.write(text.data(using: .utf8)!)
//                    } else {
//                        try text.write(to: logFile, atomically: false, encoding: .utf8)
//                    }
//
//                    var cf = ""
//                    for row in confusionMatrix {
//                        cf += "\(row)\n"
//                    }
//                    cf += "\n"
//
//                    if let fileHandle = FileHandle(forWritingAtPath: matrixFile.path) {
//                        fileHandle.seekToEndOfFile()
//                        fileHandle.write(cf.data(using: .utf8)!)
//                    } else {
//                        try cf.write(to: matrixFile, atomically: false, encoding: .utf8)
//                    }
//                }
//            }
//        }
//    }
//}

if let dir = FileManager.default.urls(for: .downloadsDirectory, in: .userDomainMask).first {
    let logFile = dir.appendingPathComponent("scores").appendingPathExtension("txt")
}
