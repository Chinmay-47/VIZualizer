import { Component, ElementRef, OnInit, ViewChild, ViewEncapsulation } from '@angular/core';
import { NavService } from './nav.service';
import { NavItem } from './nav.item';
import { HomeService } from './services/api/home.service';

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.scss'],
  encapsulation: ViewEncapsulation.None
})
export class AppComponent implements OnInit {
  @ViewChild('appDrawer') appDrawer: ElementRef | undefined;
  title = 'VIZualizer';

  navItems: NavItem[] = [
    {
      displayName: '',
      iconName: '',
      route: '',
      children: [
        {
          displayName: '',
          iconName: '',
          route: '',
          children: [
            {
              displayName: '',
              iconName: '',
              route: '',
              children: []
            }
          ]
        }
      ]
    },
  ];

  constructor(private navService: NavService, private homeService: HomeService) {
  }

  ngAfterViewInit() {
    this.navService.appDrawer = this.appDrawer;
  }

  ngOnInit() {
    this.getNavItems();
  }

  getNavItems() {
    this.navItems = []
    let obs = this.homeService.getSupervisedModels();
    obs.subscribe((response) => {
      let responseHolder: any = response;
      for (let i = 0; i < responseHolder.length; i++) {
        let sideNavData: NavItem =
        {
          displayName: '',
          iconName: 'star_rate',
          route: '',
          children: []
        }
        let innerResponseHolder: any = responseHolder[i];
        sideNavData.displayName = (("" + innerResponseHolder[0]).replace("_", " "))
        sideNavData.route = "" + innerResponseHolder[0]
        for (let j = 0; j < innerResponseHolder[1].length; j++) {
          let tempSideNavData: NavItem =
          {
            displayName: '',
            iconName: 'star_rate',
            route: '',
            children: []
          }
          tempSideNavData.displayName = (("" + innerResponseHolder[1][j]).replace("_", " "))
          tempSideNavData.route = ("" + innerResponseHolder[0] + "/" + innerResponseHolder[1][j])
          sideNavData.children?.push(tempSideNavData)
        }
        this.navItems.push(sideNavData)
      }
    })
  }
}
