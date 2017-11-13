//
//  MockRecommendEventHandler.swift
//  MovieRecTests
//
//  Created by Yujin Cho on 11/12/17.
//  Copyright © 2017 Yujin Cho. All rights reserved.
//

import Foundation
@testable import MovieRec


class MockRecommendEventHandler: RecommendModuleInterface {
    var updateViewCalled: Bool = false
    
    func updateView() {
        updateViewCalled = true
    }
}
