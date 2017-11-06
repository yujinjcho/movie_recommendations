//
//  RateViewInterface.swift
//  MovieRec
//
//  Created by Yujin Cho on 10/8/17.
//  Copyright © 2017 Yujin Cho. All rights reserved.
//

import Foundation

protocol RateViewInterface {
    func showCurrentMovie(title: String, photoUrl: String, completion: (() -> Void)?)
}
