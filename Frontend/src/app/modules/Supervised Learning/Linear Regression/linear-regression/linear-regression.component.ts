import { Component, OnInit, Pipe, PipeTransform } from '@angular/core';
import { LinearRegressionService } from 'src/app/services/api/Supervised Learning/linear-regression.service';
import { linearRegression } from './linear-regression.model';
import { DomSanitizer, SafeHtml } from '@angular/platform-browser';

@Component({
  selector: 'app-linear-regression',
  templateUrl: './linear-regression.component.html',
  styleUrls: ['./linear-regression.component.scss']
})
export class LinearRegressionComponent implements OnInit {

  constructor(private linearRegressionService: LinearRegressionService, private _sanitizer: DomSanitizer) { }

  linearRegressionObj: linearRegression = {
    learningRate: 0.001,
    epochs: 10000,
    dataPoints: 20,
    randomize: false,
    linearlyIncreasing: false
  }
  displayPlots: boolean = false;
  displayAnimationplot: boolean = false;
  imagePathDataPoints: any;
  imagePathInitialRegressionLine: any;
  imagePathRegressionComparison: any;
  imagePathCostHistory: any;
  imagePathRegressionProgression: any;
  animationPath: any;
  

  ngOnInit(): void {
  }

  linearRegression() {
    this.displayPlots = false;
    let obs = this.linearRegressionService.getPlots(this.linearRegressionObj)
    obs.subscribe((response) => {
      let responseHolder: any = response
      let dataPoints: string = ("" + responseHolder["data points"]).split('\'')[1]
      let initialLine: string = ("" + responseHolder["initial regression line"]).split('\'')[1]
      let comparison: string = ("" + responseHolder["regression line comparison"]).split('\'')[1]
      let costHistory: string = ("" + responseHolder["cost history"]).split('\'')[1]
      let progression: string = ("" + responseHolder["regression line progression"]).split('\'')[1]
      this.imagePathDataPoints = this._sanitizer.bypassSecurityTrustResourceUrl('data:image/jpg;base64,' + dataPoints);
      this.imagePathInitialRegressionLine = this._sanitizer.bypassSecurityTrustResourceUrl('data:image/jpg;base64,' + initialLine);
      this.imagePathRegressionComparison = this._sanitizer.bypassSecurityTrustResourceUrl('data:image/jpg;base64,' + comparison);
      this.imagePathCostHistory = this._sanitizer.bypassSecurityTrustResourceUrl('data:image/jpg;base64,' + costHistory);
      this.imagePathRegressionProgression = this._sanitizer.bypassSecurityTrustResourceUrl('data:image/jpg;base64,' + progression);
      this.displayPlots = true;
      this.displayAnimation()
    })
  }

  displayAnimation() {
    let obs = this.linearRegressionService.getAnimation()
    obs.subscribe((response) => {
      console.log(response)
      let responseHolder: any = response;
      this.animationPath = responseHolder["html_to_render"]
      this.displayAnimationplot = true;
    })
  }

}
